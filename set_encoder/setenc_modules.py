#####################################################################################
# Copyright (c) Juho Lee SetTransformer, ICML 2019 [GitHub set_transformer]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.modules import MetaLinear, MetaSequential, MetaModule


class MAB(MetaModule):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = MetaLinear(dim_Q, dim_V)
        self.fc_k = MetaLinear(dim_K, dim_V)
        self.fc_v = MetaLinear(dim_K, dim_V)
        if ln:
            raise NotImplementedError
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = MetaLinear(dim_V, dim_V)

    def forward(self, Q, K, params=None):
        Q = self.fc_q(Q, params=self.get_subdict(params, 'fc_q'))
        K, V = self.fc_k(K, params=self.get_subdict(params, 'fc_k')), self.fc_v(K, params=self.get_subdict(params, 'fc_v'))

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O, params=self.get_subdict(params, 'fc_o')))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(MetaModule):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, params=None):
        return self.mab(X, X, params=self.get_subdict(params, 'mab'))

class ISAB(MetaModule):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, params=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, params=self.get_subdict(params, 'mab0'))
        return self.mab1(X, H, params=self.get_subdict(params, 'mab1'))

class PMA(MetaModule):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, params=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, params=self.get_subdict(params, 'mab'))