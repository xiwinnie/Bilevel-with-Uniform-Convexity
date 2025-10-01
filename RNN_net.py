"""
Module for natural language classification.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
class NLIRNN(nn.Module):
    def __init__(
        self,
        word_embed_dim,
        encoder_dim,
        n_enc_layers,
        dpout_model,
        dpout_fc,
        fc_dim,
        n_classes,
        pool_type,
        linear_fc,
        bidirectional=False,
    ):
        super(NLIRNN, self).__init__()

        # Store settings.
        self.encoder_dim = encoder_dim
        self.n_enc_layers = n_enc_layers
        self.dpout_fc = dpout_fc
        self.fc_dim = fc_dim
        self.n_classes = n_classes
        self.linear_fc = linear_fc
        self.bidirectional = bidirectional

        # Construct encoder and classifier.
        self.encoder = RecurrentEncoder(
            n_enc_layers, word_embed_dim, encoder_dim, pool_type, dpout_model, bidirectional
        )
        self.lin = nn.Linear(4 * self.encoder_dim, self.encoder_dim)
        # self.inputdim = self.encoder_dim
        # if self.bidirectional:
        #     self.inputdim *= 2
        # if self.linear_fc:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(self.inputdim, self.fc_dim),
        #         nn.Linear(self.fc_dim, self.fc_dim),
        #         nn.Linear(self.fc_dim, self.n_classes)
        #     )
        # else:
        #     self.classifier = nn.Sequential(
        #         nn.Dropout(p=self.dpout_fc),
        #         nn.Linear(self.inputdim, self.fc_dim),
        #         nn.Tanh(),
        #         nn.Dropout(p=self.dpout_fc),
        #         nn.Linear(self.fc_dim, self.fc_dim),
        #         nn.Tanh(),
        #         nn.Dropout(p=self.dpout_fc),
        #         nn.Linear(self.fc_dim, self.n_classes),
        #     )

    def forward(self, s_1, s_2):
        u = self.encoder(s_1)
        v = self.encoder(s_2)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.lin(features)
        return output


class NLINet(nn.Module):
    def __init__(
        self,
        word_embed_dim,
        encoder_dim,
        n_enc_layers,
        dpout_model,
        dpout_fc,
        fc_dim,
        n_classes,
        pool_type,
        linear_fc,
        bidirectional=False,
    ):
        super(NLINet, self).__init__()

        # Store settings.
        self.encoder_dim = encoder_dim
        self.n_enc_layers = n_enc_layers
        self.dpout_fc = dpout_fc
        self.fc_dim = fc_dim
        self.n_classes = n_classes
        self.linear_fc = linear_fc
        self.bidirectional = bidirectional

        # Construct encoder and classifier.
        self.encoder = RecurrentEncoder(
            n_enc_layers, word_embed_dim, encoder_dim, pool_type, dpout_model, bidirectional
        )
        self.inputdim = 4 * self.encoder_dim
        if self.bidirectional:
            self.inputdim *= 2
        if self.linear_fc:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
            )

    def forward(self, s1, s2):
        u = self.encoder(s1)
        v = self.encoder(s2)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output


class RNN(nn.Module):
    def __init__(
        self,
        word_embed_dim,
        encoder_dim,
        n_enc_layers,
        dpout_model,
        dpout_fc,
        fc_dim,
        n_classes,
        pool_type,
        linear_fc,
        bidirectional=False,
    ):
        super(RNN, self).__init__()

        # Store settings.
        self.encoder_dim = encoder_dim
        self.n_enc_layers = n_enc_layers
        self.dpout_fc = dpout_fc
        self.fc_dim = fc_dim
        self.n_classes = n_classes
        self.linear_fc = linear_fc
        self.bidirectional = bidirectional

        # Construct encoder and classifier.
        self.encoder = RecurrentEncoder(
            n_enc_layers, word_embed_dim, encoder_dim, pool_type, dpout_model, bidirectional
        )

        self.inputdim =  self.encoder_dim
        if self.bidirectional:
            self.inputdim *= 2
        if self.linear_fc:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
            )

    def forward(self, s1, s2=None):
        if s2 is not None:
            u = self.encoder(s1)
            v = self.encoder(s2)
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        else:
            features = self.encoder(s1)
        output = self.classifier(features)
        return output

class RecurrentEncoder(nn.Module):

    def __init__(
        self, n_enc_layers, word_embed_dim, encoder_dim, pool_type, dpout_model, bidirectional, rnn=True
    ):
        super(RecurrentEncoder, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.word_embed_dim = word_embed_dim
        self.encoder_dim = encoder_dim
        self.pool_type = pool_type
        self.dpout_model = dpout_model
        self.bidirectional = bidirectional
        self.rnn = rnn

        net_cls = nn.RNN if self.rnn else nn.LSTM
        self.encoder = net_cls(
            self.word_embed_dim,
            self.encoder_dim,
            self.n_enc_layers,
            bidirectional=bidirectional,
            dropout=self.dpout_model,
        )

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(self.encoder.bias_hh_l0.device)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        self.encoder.flatten_parameters()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        sent_len, idx_sort = sent_len.copy(), idx_sort.copy()
        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.encoder(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()  if self.is_cuda() \
            else Variable(torch.FloatTensor(sent_len)).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

class RecurrentEncoder_(nn.Module):

    def __init__(
        self, n_enc_layers, word_embed_dim, encoder_dim, pool_type, dpout_model, bidirectional, rnn=True
    ):
        super(RecurrentEncoder_, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.word_embed_dim = word_embed_dim
        self.encoder_dim = encoder_dim
        self.pool_type = pool_type
        self.dpout_model = dpout_model
        self.bidirectional = bidirectional
        self.rnn = rnn

        net_cls = nn.RNN if self.rnn else nn.LSTM
        self.encoder = net_cls(
            self.word_embed_dim,
            self.encoder_dim,
            self.n_enc_layers,
            bidirectional=bidirectional,
            dropout=self.dpout_model,
        )

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(self.encoder.bias_hh_l0.device)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        self.encoder.flatten_parameters()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        sent_len, idx_sort = sent_len.copy(), idx_sort.copy()
        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.encoder(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()  if self.is_cuda() \
            else Variable(torch.FloatTensor(sent_len)).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb
