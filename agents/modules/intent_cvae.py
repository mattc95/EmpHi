import math
import copy
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import linalg as LA
from .gru import GRUModel

class Encoder(nn.Module):

    def __init__(self, num_layers, embeddings, hidden_size, padding_idx):
        """
            Encoder of EmpHi
            Including Bi-GRU
        """
        # must call super on all nn.Modules.
        super().__init__()

        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_index = padding_idx
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, seq, seq_len):

        batchsize = seq.size(0)
        seq_emb = self.embeddings(seq)

        mask = (seq != self.padding_index)

        pack_seq = rnn_utils.pack_padded_sequence(seq_emb, seq_len, batch_first=True, enforce_sorted=False)

        # seq_hidden = [2*num_layers, batch, hidden_size]

        pack_seq_out, seq_hidden = self.gru(pack_seq)

        seq_out, _ = rnn_utils.pad_packed_sequence(pack_seq_out, batch_first=True)

        seq_out = seq_out.view(batchsize, -1, 2, self.hidden_size).mean(dim=-2)
        seq_hidden = seq_hidden.view(2, self.num_layers, batchsize, self.hidden_size).mean(dim=0)

        # seq_out    = [batch, seq_len, hidden_size]
        # seq_hidden = [num_layers, batch, hidden_size]
        return seq_out, seq_hidden, mask


class EmotionIntentDecoder(nn.Module):

    def __init__(self, implicit, implicit_dynamic, explicit, dropout, num_layers, embeddings, hidden_size, padding_idx):
        """
            Decoder of EmpHi
            Including GRU, Implicit Gate for Intent, Implicit Gate for Emotion
        """
        super().__init__()
        self.embeddings = embeddings
        self.num_layers = num_layers

        self.implicit = implicit
        self.implicit_dynamic = implicit_dynamic
        self.explicit = explicit

        self.dropout = dropout
        self.drop_net = nn.Dropout(dropout)

        num = 2
        if implicit:
            num += 2

        self.gru = nn.GRU(
            input_size=num*hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )           

        self.intent_read_gate = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size, bias=False),
            nn.Sigmoid(),
        )

        self.emotion_read_gate = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, seq, emotion_emb, intent_emb, context, state, mask):

        bsz = seq.size(0)
        seq_emb = self.embeddings(seq)

        """
            context attention
        """
        
        context_attention = context.matmul(state[-1].unsqueeze(dim=-1)).squeeze(dim=-1)

        context_attention = context_attention.masked_fill_(~mask, -1e9)
        # [bsz, seq_len]
        context_attention = F.softmax(context_attention, dim=-1)
        context_value = context_attention.unsqueeze(dim=-1) * context
        context_value = context_value.sum(dim=-2) 

        input = torch.cat([seq_emb, context_value], dim=-1)

        if self.implicit is True and self.dropout != 0:
            intent_emb = self.drop_net(intent_emb)

        if self.implicit is True:
            if self.implicit_dynamic is True:
            
                read_input = torch.cat([state[-1], seq_emb, context_value], dim=-1)
                emotion_gate = self.emotion_read_gate(read_input)
                emotion_emb = emotion_gate * emotion_emb

                read_input = torch.cat([state[-1], seq_emb, context_value], dim=-1)
                intent_gate = self.intent_read_gate(read_input)
                intent_emb = intent_gate * intent_emb

            input = torch.cat([input, emotion_emb, intent_emb], dim=-1)

        input = input.unsqueeze(dim=1)

        # get the new output and decoder incremental state
        # output = [batch, 1, hidden_size]
        # state  = [num_layers, batch, hidden_size]
        output, state = self.gru(input, state)

        return output, state


class IntentCVAEGRUModel(GRUModel):
    
    """
        The Seq2seq model of EmpHi
        Including Encoder, Intent Predictor, Emotion Classifier, Decoder, Copy Mechanism

    """
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)

        hidden_size = opt['embedding_size']
        
        self.hidden_size = hidden_size
        
        self.emotion_embeddings = nn.Embedding(32, hidden_size)
        self.intent_embeddings = nn.Embedding(9, hidden_size)

        self.emotion_prediction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 32, bias=False)
        )

        self.intent_prediction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 9, bias=False)
        )

        self.copy_gate = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False), 
            nn.Sigmoid(),
        )
        
        self.explicit = opt['explicit']

        self.encoder = Encoder(self.num_layers, self.embeddings, hidden_size, dictionary[dictionary.null_token])
        self.decoder = EmotionIntentDecoder \
            (opt['implicit'], opt['implicit_dynamic'], opt['explicit'], opt['dropout'], self.num_layers, self.embeddings, hidden_size, dictionary[dictionary.null_token])

    def output(self, output, state, intent_index, intent_knowledge):
        """
        Perform the final output -> logits transformation.
        """

        bsz = output.shape[0]
        
        copy_rate = self.copy_gate(state[-1])
      
        logit = F.linear(output, self.embeddings.weight)
      
        if self.explicit:
            prob = F.softmax(logit, dim=-1)

            copy_logit = intent_knowledge.matmul(output.unsqueeze(dim=-1)).squeeze(dim=-1)    
            copy_prob = F.softmax(copy_logit, dim=-1)

            prob = ((1 - copy_rate) * prob).scatter_add \
                (dim=1, index=intent_index, src=copy_rate * copy_prob)

            logit = torch.log(prob)

        return logit, copy_rate.squeeze(dim=-1)

    def decode_forced(self, emotion, intent, intent_knowledge, encoder_state, ys):

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
        copy_rate = torch.zeros(bsz, res_len).float().to(device)
        context, state, mask = encoder_state

        emotion_emb = emotion.matmul(self.emotion_embeddings.weight)

        intent_emb = self.intent_embeddings(intent)

        intent_1gram = intent_knowledge
        intent_index = intent_1gram.index_select(dim=0, index=intent) 
        
        # [9, 30, hidden_size]
        intent_knowledge = self.embeddings(intent_1gram.squeeze(dim=-1))

        # [bsz, 60, hidden_size]
        intent_knowledge = intent_knowledge.index_select(dim=0, index=intent)

        for t in range(res_len):

            out, state = \
            self.decoder(inputs, emotion_emb, intent_emb, context, state, mask)

            logit[:, t], copy_rate[:, t] = self.output(out.squeeze(dim=1), state, intent_index, intent_knowledge)
            # Place predictions in a tensor holding predictions for each token
            pred[:, t] = logit[:, t].argmax(-1)

            inputs = ys[:, t]

        return logit, pred, copy_rate

    def decode_greedy(self, emotion, intent, intent_knowledge, encoder_states, max_len=64):

        context, state, mask = encoder_states
        bsz = context.size(0)

        inputs = self.START.detach().expand(bsz)

        logits = []

        emotion_emb = emotion.matmul(self.emotion_embeddings.weight)

        intent_emb = self.intent_embeddings(intent)

        intent_1gram = intent_knowledge
        intent_index = intent_1gram.index_select(dim=0, index=intent) 
        
        # [9, 30, hidden_size]
        intent_knowledge = self.embeddings(intent_1gram.squeeze(dim=-1))

        # [bsz, 60, hidden_size]
        intent_knowledge = intent_knowledge.index_select(dim=0, index=intent)

        for t in range(max_len):

            out, state = \
                self.decoder(inputs, emotion_emb, intent_emb, context, state, mask)

            logit, _ = self.output(out.squeeze(dim=1), state, intent_index, intent_knowledge)
            # Place predictions in a tensor holding predictions for each token
            inputs = logit.argmax(-1)
            logits.append(logit.unsqueeze(dim=1))

        logits = torch.cat(logits, dim=1)  # (bsz, seqlen, vocab)
        _, preds = logits.max(dim=2)

        return preds