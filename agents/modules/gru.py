import math
import copy
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):

    def __init__(self, num_layers, embeddings, hidden_size, padding_idx):
        """
            Bi-GRU Encoder
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

class Decoder(nn.Module):


    def __init__(self, num_layers, embeddings, hidden_size, padding_idx):
        """
            GRU Decoder
        """
        super().__init__()
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=2*hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    def forward(self, seq, context, state, mask):

        batchsize = seq.size(0)
        seq_emb = self.embeddings(seq)
        s = state.view(self.num_layers, batchsize, self.hidden_size).mean(dim=0)
        att = F.softmax(context.matmul(s.unsqueeze(-1)).squeeze(-1), dim=-1)
        att = att * mask

        att_context = (att.unsqueeze(-1) * context).sum(dim=-2)
        input = torch.cat([seq_emb, att_context], dim=-1).unsqueeze(dim=1)

        # get the new output and decoder incremental state
        # output = [batch, 1, hidden_size]
        # state  = [num_layers, batch, hidden_size]
        output, state = self.gru(input, state)

        return output, state


class GRUModel(nn.Module):

    """
        Seq2seq GRU model
    """
    def __init__(self, opt, dictionary):
        super().__init__()
        self.NULL_IDX = dictionary[dictionary.null_token]
        self.END_IDX = dictionary[dictionary.end_token]
        self.START_IDX = dictionary[dictionary.start_token]
        self.register_buffer('START', torch.LongTensor([self.START_IDX]))

        hidden_size = opt['embedding_size']
        self.max_res_len = opt['label_truncate']
        self.vocab_size = len(dictionary)
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.num_layers = opt['num_layers']
        self.encoder = Encoder(self.num_layers, self.embeddings, hidden_size, dictionary[dictionary.null_token])
        self.decoder = Decoder(self.num_layers, self.embeddings, hidden_size, dictionary[dictionary.null_token])

    def output(self, decoder_output):
        """
        Perform the final output -> logits transformation.
        """
        return F.linear(decoder_output, self.embeddings.weight)

    def decode_forced(self, encoder_state, ys):

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

        for t in range(res_len):

            out, state = self.decoder(inputs, context, state, mask)

            logit[:, t] = self.output(out).squeeze(dim=1)
            # Place predictions in a tensor holding predictions for each token
            pred[:, t] = logit[:, t].argmax(-1)

            inputs = ys[:, t]

        return logit, pred

    def decode_greedy(self, encoder_states, max_len=64):

        context, state, mask = encoder_states
        bsz = context.size(0)

        inputs = self.START.detach().expand(bsz)

        logits = []

        for t in range(max_len):
            output, state = self.decoder(inputs, context, state, mask)
            logit = self.output(output)
            # Place predictions in a tensor holding predictions for each token
            inputs = logit.argmax(-1).squeeze(1)
            logits.append(logit)

        logits = torch.cat(logits, dim=1)  # (bsz, seqlen, vocab)
        _, preds = logits.max(dim=2)

        return logits, preds
