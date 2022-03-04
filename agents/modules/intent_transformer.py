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


class ConditionalTransformer(TransformerGeneratorModel):
    """
    Implements a full transformer generator model, with pragmatic self-consciousness.
    """

    def __init__(self, opt, dictionary):

        super().__init__(opt, dictionary)
        self.fp16 = opt['fp16']
        self.max_len = opt['label_truncate']
        self.intent_emb = nn.Embedding(41, opt['embedding_size'])
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
