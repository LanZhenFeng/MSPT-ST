import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.OpenSTL_Modules import STconvert


class Model(nn.Module):
    r"""SwinLSTM 

    Implementation of `SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin
    Transformer and LSTM <http://arxiv.org/abs/2308.09891>`_.

    """

    def __init__(self, configs, window_size=4):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # curriculum learning strategy
        self.cls = configs.curriculum_learning_strategy # RSS, SS or None
        curriculum_learning_strategies = ['rss', 'ss', 'none']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 'none']"

        self.ST = STconvert(img_size=[configs.height, configs.width], patch_size=configs.patch_size, in_chans=configs.enc_in, embed_dim=configs.d_model, depths=configs.e_layers, num_heads=configs.n_heads, window_size=window_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        
        mask_true = rearrange(mask_true, 'b t h w c -> b t c h w')

        x_enc = torch.cat((x_enc, x_dec[:, -self.pred_len:]), dim=1)

        x_ts = rearrange(x_enc, 'b t h w c -> b t c h w')

        states = None

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

            x_gen, states = self.ST(x_t, states)
            if t >= self.seq_len - 1:
                predictions.append(x_gen)

        dec_out = torch.stack(predictions, dim=1)
        
        dec_out = rearrange(dec_out, 'b t c h w -> b t h w c')
        
        return dec_out, None