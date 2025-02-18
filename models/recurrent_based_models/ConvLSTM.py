import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.OpenSTL_Modules import ConvLSTMCell


class Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, configs, filter_size=5, stride=1, layer_norm=False):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        self.num_hidden = configs.d_model
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm

        # curriculum learning strategy
        self.cls = configs.curriculum_learning_strategy # RSS, SS or None
        curriculum_learning_strategies = ['rss', 'ss', 'none']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 'none']"

        cell_list = []
        self.d_patch = configs.patch_size * configs.patch_size * configs.enc_in
        height = configs.height // configs.patch_size
        width = configs.width // configs.patch_size
        for i in range(self.num_layers):
            in_channel = self.d_patch if i == 0 else self.num_hidden
            cell_list.append(ConvLSTMCell(in_channel, self.num_hidden, height, width, self.filter_size, self.stride, self.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(self.num_hidden, self.d_patch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward_one_time_step(self, x_t, h_t, c_t):
        h_t[0], c_t[0] = self.cell_list[0](x_t, h_t[0], c_t[0])
        for i in range(1, self.num_layers):
            h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        
        mask_true = rearrange(mask_true, 'b t h w c -> b t c h w')

        x_enc = torch.cat((x_enc, x_dec[:, -self.pred_len:]), dim=1)

        # patching
        x_ts = rearrange(x_enc, 'b t (p1 h) (p2 w) c -> b t (c p1 p2) h w', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        h_t = []    # hidden states
        c_t = []    # cell states

        # Initialize hidden states and cell states
        B, T, D, H, W = x_ts.shape
        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden, H, W], device=x_ts.device)
            h_t.append(zeros)
            c_t.append(zeros)

        predictions = []    # save the predictions

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

            self.forward_one_time_step(x_t, h_t, c_t)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            if t >= self.seq_len - 1:
                predictions.append(x_gen)

        dec_out = torch.stack(predictions, dim=1)
        
        # patch recovery
        dec_out = rearrange(dec_out, 'b t (c p1 p2) h w -> b t (p1 h) (p2 w) c', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        return dec_out, None