import math
import copy
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.agents.transformer.modules import \
    TransformerGeneratorModel, TransformerDecoder, TransformerDecoderLayer
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

class MultiTaskTransformer(TransformerGeneratorModel):
    """
    Implements a full transformer generator model, with pragmatic self-consciousness.
    """

    def __init__(self, opt, dictionary):

        super().__init__(opt, dictionary)
        self.classification = nn.Linear(opt['embedding_size'], 32, bias=False)

class ConditionalTransformer(TransformerGeneratorModel):
    """
    Implements a full transformer generator model, with pragmatic self-consciousness.
    """

    def __init__(self, opt, dictionary):

        super().__init__(opt, dictionary)
        self.fp16 = opt['fp16']
        self.max_len = opt['label_truncate']
        self.intent_emotion_emb = nn.Embedding(41, opt['embedding_size'])
        self.prior = nn.Sequential(
            nn.Linear(opt['embedding_size'], opt['embedding_size']),
            nn.ReLU(),
            nn.Linear(opt['embedding_size'], 41),
            nn.Softmax()
        )

        self.share_decoder = TransformerDecoderLayer(
            n_heads=opt['n_heads'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            activation=opt['activation'],
            variant=opt['variant'],
        )
        self.meta_decoder = TransformerDecoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=self.embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=opt['n_positions'],
            activation=opt['activation'],
            variant=opt['variant'],
        )
        self.multi_decoders = nn.ModuleList()
        for i in range(41):
            self.multi_decoders.append(
                TransformerDecoderLayer(
                    n_heads=opt['n_heads'],
                    embedding_size=opt['embedding_size'],
                    ffn_size=opt['ffn_size'],
                    dropout=opt['dropout'],
                    attention_dropout=opt['attention_dropout'],
                    relu_dropout=opt['relu_dropout'],
                    activation=opt['activation'],
                    variant=opt['variant'],
                )
            )

    def decode_forced(self, condition, encoder_states, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        seq_len = inputs.size(1)
        positions = inputs.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        inputs_embedding = self.meta_decoder.forward_embedding(inputs, positions)

        latent, _ = self.share_decoder(inputs_embedding, encoder_states[0], encoder_states[1], incr_state={})

        for i in range(41):
            expert_latent, _ = self.multi_decoders[i](inputs_embedding, encoder_states[0], encoder_states[1], incr_state={})
            latent += condition[:, i].unsqueeze(-1).unsqueeze(-1) * expert_latent

        latent, _ = self.meta_decoder.forward_layers\
            (latent, encoder_states[0], encoder_states[1], incr_state={})

        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def decode_greedy(self, condition, encoder_states):

        bsz = encoder_states[0].size(0)

        inputs = self.START.detach().expand(bsz, 1)

        share_state = None
        multi_states = [{} for i in range(41)]
        meta_state = {}
        logits = []

        for t in range(self.max_len):

            seq_len = inputs.size(1)
            positions = inputs.new(seq_len).long()
            positions = torch.arange(seq_len, out=positions).unsqueeze(0)

            if share_state is not None:
                # We're doing incremental decoding, so select only the most recent position
                inputs = inputs[:, -1:]
                positions = positions[:, -1:]

            inputs_embedding = self.meta_decoder.forward_embedding(inputs, positions)

            latent, share_state = self.share_decoder(inputs_embedding,
                                                     encoder_states[0], encoder_states[1], share_state)
            for i in range(41):
                expert_latent, multi_states[i] = \
                    self.multi_decoders[i](inputs_embedding,
                                           encoder_states[0], encoder_states[1], multi_states[i])
                latent += condition[:, i].unsqueeze(-1).unsqueeze(-1) * expert_latent

            latent, meta_state = self.meta_decoder.forward_layers \
                (latent, encoder_states[0], encoder_states[1], meta_state)

            logit = self.output(latent).select(dim=1, index=-1).unsqueeze(1)
            logits.append(logit)

            next_token = logit.max(dim=2)[1].clone().detach()  # next input is current predicted output idx

            # update inputs for next timestep
            inputs = torch.cat((inputs, next_token), dim=1)

        logits = torch.cat(logits, dim=1)  # (bsz, seqlen, vocab)
        _, preds = logits.max(dim=2)

        return logits, preds




class MutualGRUModel(nn.Module):
    """
    Implements a full transformer generator model, with pragmatic self-consciousness.
    """

    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)

        self.alpha = opt['alpha']
        self.theta = opt['theta']
        self.num_samples = opt['num_samples']
        # self.prior = opt['prior']
        self.truth_id = -1
        self.dict = dictionary

    def _initialize_prior(self, bsz):
        """
        initialize the world prior with a uniform distribution
        """
        num_samples = self.num_samples
        ones = torch.ones(1, 1, num_samples, requires_grad=False).cuda()
        prior = torch.log(ones / num_samples).repeat(bsz, 1, 1).detach()

        return prior

    def _mutual_information_maximization(self, s0_t, prior, length):
        """
        run pragmatic reasoning with the base speaker and its imaginary listener
        """

        vocab_size = self.embeddings.num_embeddings

        # log-scale
        # [bsz*num, vocab_size]
        log_token_prob = F.log_softmax(s0_t, dim=1)

        # [bsz, num, vocab]
        log_token_prob = log_token_prob.view(-1, self.num_samples, vocab_size)

        # S_0 for the actual given persona (bsz, vocab)
        truth_log_token_prob = log_token_prob.select(1, self.truth_id)  # target persona is always index 0

        # S_0 for L_0
        # (bsz, vocab, world_cardinality)
        log_token_prob = log_token_prob.transpose(dim0=1, dim1=2).contiguous()

        # L_0 \propto S_0 * p(i)
        # worldprior should be broadcasted to all the tokens
        # (bsz, vocab, world_cardinality)

        theta = min((length + 1) * 0.1, 1)
        # theta = 0.5
        log_posterior = (theta * log_token_prob + prior.detach()) - \
                        torch.logsumexp(theta * log_token_prob + prior.detach(), 2, keepdim=True)

        # (bsz, vocab)
        truth_log_posterior = log_posterior.select(2, self.truth_id)  # target persona is always index 0

        mutual_log_prob = (1 - self.alpha) * truth_log_token_prob + self.alpha * truth_log_posterior
        # Normalization
        mutual_log_prob = mutual_log_prob - torch.logsumexp(mutual_log_prob, 1, keepdim=True)

        return mutual_log_prob, log_posterior, log_token_prob

    def decode_forced(self, encoder_state, ys):
        """
        faster teacher-forced decoding with pragmatic self-consciousness
        """
        context, state, mask = encoder_state
        bsz = ys.size(0)
        s_bsz = context.size(0)
        device = ys.device
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self.START.detach().expand(s_bsz).to(device)
        res_len = ys.size(1)
        prior = self._initialize_prior(bsz)
        logits = []
        mutual_logits = []
        log_posteriors = []

        for t in range(res_len):

            out, state = self.decoder(inputs, context, state, mask)

            logit = self.output(out).squeeze(dim=1)

            logits.append(logit.view(bsz, self.num_samples, -1)[:, self.truth_id, :].unsqueeze(dim=1))
            mutual_log_prob, log_posterior, log_token_prob = \
                self._mutual_information_maximization(logit, prior, length=t)

            mutual_logits.append(mutual_log_prob.unsqueeze(dim=1))

            prior = torch.gather(input=log_posterior, dim=1,
                                 index=ys[:, t].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_samples))
            # print(ys[0, t])
            # print(self.dict[ys[0, t].item()])
            # print(torch.exp(log_posterior[0, ys[0, t]]))
            # print(torch.exp(prior))
            log_posteriors.append(prior[:, :, self.truth_id])

            inputs = ys[:, t].unsqueeze(-1).expand(-1, self.num_samples).contiguous().view(-1)


        logits = torch.cat(logits, dim=1)
        _, preds = logits.max(dim=2)
        mutual_logits = torch.cat(mutual_logits, dim=1)
        _, mutual_preds = mutual_logits.max(dim=2)
        log_posteriors = torch.cat(log_posteriors, dim=1)

        return logits, preds, mutual_logits, mutual_preds, log_posteriors

    def decode_greedy(self, encoder_state, max_len=32):
        """
        greedy decoding with pragmatic self-consciousness
        """
        context, state, mask = encoder_state
        bsz = context.size(0) // self.num_samples
        s_bsz = context.size(0)
        device = context.device

        inputs = self.START.detach().expand(s_bsz).to(device)
        prior = self._initialize_prior(bsz)
        logits = []
        mutual_logits = []
        mutual_preds = []
        log_posteriors = []

        for t in range(max_len):

            out, state = self.decoder(inputs, context, state, mask)

            logit = self.output(out).squeeze(dim=1)
            logits.append(logit.view(bsz, self.num_samples, -1)[:, self.truth_id, :].unsqueeze(dim=1))
            mutual_log_prob, log_posterior, log_token_prob = \
                self._mutual_information_maximization(logit, prior, length=t)

            mutual_logits.append(mutual_log_prob.unsqueeze(dim=1))

            preds = mutual_log_prob.argmax(dim=-1)
            mutual_preds.append(preds.unsqueeze(-1))

            prior = torch.gather(input=log_posterior, dim=1,
                                 index=preds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_samples))

            # print(self.dict[preds.item()])
            # print(torch.exp(log_token_prob[0, preds]))
            # print(torch.exp(log_posterior[0, preds]))
            # print(torch.exp(prior))

            log_posteriors.append(prior[:, :, self.truth_id])
            inputs = preds.unsqueeze(-1).expand(-1, self.num_samples).contiguous().view(-1)


        logits = torch.cat(logits, dim=1)
        _, preds = logits.max(dim=2)

        mutual_logits = torch.cat(mutual_logits, dim=1)
        mutual_preds = torch.cat(mutual_preds, dim=1)
        log_posteriors = torch.cat(log_posteriors, dim=1)

        return logits, preds, mutual_logits, mutual_preds, log_posteriors



class EmotionIntentDecoder(nn.Module):
    """
    Basic example decoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size. Decoder allows for incremental decoding by ingesting the
    current incremental state on each forward pass.
    Pay particular note to the ``forward``.
    """

    def __init__(self, num_layers, embeddings, hidden_size, padding_idx):
        """
        Initialization.
        Arguments here can be used to provide hyperparameters.
        """
        super().__init__()
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=4*hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    def forward(self, seq, emotion_emb, intent_emb, context, state, mask):

        batchsize = seq.size(0)
        seq_emb = self.embeddings(seq)
        s = state.view(self.num_layers, batchsize, self.hidden_size).mean(dim=0)
        att = F.softmax(context.matmul(s.unsqueeze(-1)).squeeze(-1), dim=-1)
        att = att * mask

        att_context = (att.unsqueeze(-1) * context).sum(dim=-2)
        input = torch.cat([seq_emb, att_context, emotion_emb, intent_emb], dim=-1).unsqueeze(dim=1)

        # get the new output and decoder incremental state
        # output = [batch, 1, hidden_size]
        # state  = [num_layers, batch, hidden_size]
        output, state = self.gru(input, state)

        return output, state


class IntentCVAEGRUModel(nn.Module):

    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)

        hidden_size = opt['embedding_size']
        
        
        self.emotion_embeddings = nn.Embedding(32, hidden_size)
        self.intent_embeddings = nn.Embedding(9, hidden_size)

        self.ffn = nn.Linear(2*hidden_size, hidden_size)

        self.emotion_prediction = nn.Sequential(
            nn.Linear(self.num_layers*hidden_size, hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, 32)
        )

        self.intent_prediction = nn.Sequential(
            nn.Linear(self.num_layers*hidden_size, hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, 9)
        )

        self.decoder = EmotionIntentDecoder(self.num_layers, self.embeddings, hidden_size, dictionary[dictionary.null_token])

    def decode_forced(self, emotion, intent, encoder_state, ys):

        bsz = ys.size(0)
        device = ys.device
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )

        inputs = self.START.detach().expand(bsz)
        res_len = ys.size(1)

        logit = torch.zeros(bsz, res_len, self.vocab_size).to(device)
        pred = torch.zeros(bsz, res_len).long().to(device)
        context, state, mask = encoder_state

        emotion_emb = emotion.float().matmul(self.emotion_embeddings.weight)
        intent_emb = intent.matmul(self.intent_embeddings.weight)

        for t in range(res_len):

            out, state = self.decoder(inputs, emotion_emb, intent_emb, context, state, mask)

            logit[:, t] = self.output(out).squeeze(dim=1)
            # Place predictions in a tensor holding predictions for each token
            pred[:, t] = logit[:, t].argmax(-1)

            inputs = ys[:, t]

        return logit, pred

    def decode_greedy(self, emotion, intent, encoder_states, max_len=64):

        context, state, mask = encoder_states
        bsz = context.size(0)

        inputs = self.START.detach().expand(bsz)

        logits = []

        emotion_emb = emotion.float().matmul(self.emotion_embeddings.weight)
        intent_emb = intent.matmul(self.intent_embeddings.weight)

        for t in range(max_len):

            out, state = self.decoder(inputs, emotion_emb, intent_emb, context, state, mask)
            logit = self.output(out)
            # Place predictions in a tensor holding predictions for each token
            inputs = logit.argmax(-1).squeeze(1)
            logits.append(logit)

        logits = torch.cat(logits, dim=1)  # (bsz, seqlen, vocab)
        _, preds = logits.max(dim=2)

        return logits, preds