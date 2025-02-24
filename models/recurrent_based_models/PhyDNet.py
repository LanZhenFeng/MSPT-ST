import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
# add layers to the system path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layers.OpenSTL_Modules import PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN, K2M


class Model(nn.Module):
    r"""PhyDNet Model

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # curriculum learning strategy
        self.cls = configs.curriculum_learning_strategy # RSS, SS or None
        curriculum_learning_strategies = ['rss', 'ss', 'none']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 'none']"

        patch_size = configs.patch_size if configs.patch_size in [2, 4] else 4
        height = configs.height // configs.patch_size
        width = configs.width // configs.patch_size
        input_shape = (height, width)

        self.phycell = PhyCell(input_shape=input_shape, input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device='cuda')
        self.convcell = PhyD_ConvLSTM(input_shape=input_shape, input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device='cuda')
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell, in_channel=configs.enc_in, patch_size=patch_size)
        self.k2m = K2M([7,7])

        self.constraints = self._get_constraints()

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7), device='cuda')
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        
        mask_true = rearrange(mask_true, 'b t h w c -> b t c h w')

        x_enc = torch.cat((x_enc, x_dec[:, -self.pred_len:]), dim=1)

        x_ts = rearrange(x_enc, 'b t h w c -> b t c h w')
        
        predictions = []

        for t in range(self.seq_len + self.pred_len - 1):
            if self.cls == "rss":
                # reverse schedule sampling
                if t == 0:
                    x_t = x_ts[:, t]
                else:
                    x_t = mask_true[:, t - 1] * x_ts[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif self.cls == "ss":
                # schedule sampling
                if t < self.seq_len:
                    x_t = x_ts[:, t]
                else:
                    x_t = mask_true[:, t - self.seq_len] * x_ts[:, t] + (1 - mask_true[:, t - self.seq_len]) * x_gen
            else:
                # no curriculum learning strategy
                if t < self.seq_len:
                    x_t = x_ts[:, t]
                else:
                    x_t = x_gen

            _, _, x_gen, _, _ = self.encoder(x_t, (t==0))
            if t >= self.seq_len - 1:
                predictions.append(x_gen)

        dec_out = torch.stack(predictions, dim=1)
        dec_out = rearrange(dec_out, 'b t c h w -> b t h w c')

        # auxiliary_loss
        auxiliary_loss = 0
        for b in range(self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
            m = self.k2m(filters.double()).float()
            auxiliary_loss += F.mse_loss(m, self.constraints)
        auxiliary_loss = auxiliary_loss / self.encoder.phycell.cell_list[0].input_dim
        
        return dec_out, auxiliary_loss
    