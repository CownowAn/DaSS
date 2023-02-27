###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import torch.nn as nn
from set_encoder.setenc_modules import *
from networks.modules import MetaLinear, MetaSequential, MetaModule


class SetPool(MetaModule):
    def __init__(self, dim_input, num_outputs, dim_output,
        num_inds=32, dim_hidden=128, num_heads=4, ln=False, mode=None):
        super(SetPool, self).__init__()
        if 'sab' in mode: # [32, 400, 128]
            self.enc = MetaSequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),  # SAB?
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        else: # [32, 400, 128]
            self.enc = MetaSequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),  # SAB?
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        if 'PF' in mode: #[32, 1, 501]
            self.dec = MetaSequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            MetaLinear(dim_hidden, dim_output))
        elif 'P' in mode:
            self.dec = MetaSequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln))
        else: #torch.Size([32, 1, 501])
            self.dec = MetaSequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln), # 32 1 128
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            MetaLinear(dim_hidden, dim_output))
    # "", sm, sab, sabsm
    def forward(self, X, params=None):
        x1 = self.enc(X, params=self.get_subdict(params, 'enc'))
        x2 = self.dec(x1, params=self.get_subdict(params, 'dec'))
        return x2