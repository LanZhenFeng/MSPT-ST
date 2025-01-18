import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.OpenSTL_Modules import SpatioTemporalLSTMCell, MIMBlock, MIMN    


class Model(nn.Module):
    r"""MIM Model

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

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
        self.cls = configs.curriculum_learning_strategy # RSS, SS or Standard
        curriculum_learning_strategies = ['rss', 'ss', 's']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 's']"

        stlstm_layer, stlstm_layer_diff = [], []
        self.d_patch = configs.patch_size * configs.patch_size * configs.enc_in
        height = configs.height // configs.patch_size
        width = configs.width // configs.patch_size

        for i in range(self.num_layers):
            in_channel = self.d_patch if i == 0 else self.num_hidden
            if i < 1:
                stlstm_layer.append(SpatioTemporalLSTMCell(in_channel, self.num_hidden, height, width, self.filter_size, self.stride, self.layer_norm))
            else:
                stlstm_layer.append(MIMBlock(in_channel, self.num_hidden, height, width, self.filter_size, self.stride, self.layer_norm))
        
        for i in range(self.num_layers-1):
            stlstm_layer_diff.append(MIMN(self.num_hidden, self.num_hidden, height, width, self.filter_size, self.stride, self.layer_norm))
        
        self.stlstm_layer = nn.ModuleList(stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(stlstm_layer_diff)

        self.conv_last = nn.Conv2d(self.num_hidden, self.d_patch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward_at_time_step_0(self, x_t, h_t, c_t, st_memory, hidden_state_diff):
        h_t[0], c_t[0], st_memory = self.stlstm_layer[0](x_t, h_t[0], c_t[0], st_memory)

        self.stlstm_layer_diff[0](torch.zeros_like(h_t[0]), None, None)
        h_t[1], c_t[1], st_memory = self.stlstm_layer[1](h_t[0], hidden_state_diff[0], h_t[1], c_t[1], st_memory)

        for i in range(2, self.num_layers):
            self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)
            h_t[i], c_t[i], st_memory = self.stlstm_layer[i](h_t[i - 1], hidden_state_diff[i-1], h_t[i], c_t[i], st_memory)
    
    def forward_one_time_step(self, x_t, h_t, c_t, st_memory, hidden_state_diff, cell_state_diff):
        preh = h_t[0]
        h_t[0], c_t[0], st_memory = self.stlstm_layer[0](x_t, h_t[0], c_t[0], st_memory)

        hidden_state_diff[0], cell_state_diff[0] = self.stlstm_layer_diff[0](h_t[0] - preh, hidden_state_diff[0], cell_state_diff[0])
        h_t[1], c_t[1], st_memory = self.stlstm_layer[1](h_t[0], hidden_state_diff[0], h_t[1], c_t[1], st_memory)

        for i in range(2, self.num_layers):
            hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
            h_t[i], c_t[i], st_memory = self.stlstm_layer[i](h_t[i - 1], hidden_state_diff[i-1], h_t[i], c_t[i], st_memory)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        if kwargs.get('mask_true', None) is not None:
            mask_true = kwargs['mask_true']
        else:
            if self.cls in ['rss', 'ss']:
                raise ValueError("mask_true is required for RSS and SS")
            mask_true = torch.zeros(x_enc.shape[0], self.pred_len, x_enc.shape[2], x_enc.shape[3], x_enc.shape[4], device=x_enc.device)
        
        mask_true = rearrange(mask_true, 'b t h w c -> b t c h w')
        
        if kwargs.get('batch_y', None) is not None:
            batch_y = kwargs['batch_y']
            x_enc = torch.cat([x_enc, batch_y[:, -self.seq_len:]], dim=1)
        else:
            if self.cls in ['rss', 'ss']:
                raise ValueError("batch_y is required for RSS and SS")

        # patching
        x_ts = rearrange(x_enc, 'b t (p1 h) (p2 w) c -> b t (c p1 p2) h w', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        h_t = []    # hidden states
        c_t = []    # cell states
        hidden_state_diff = []   # hidden state differences
        cell_state_diff = []    # cell state differences
        # Initialize hidden states and cell states
        B, T, D, H, W = x_ts.shape
        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden, H, W], device=x_ts.device)
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        st_memory = torch.zeros([B, self.num_hidden, H, W], device=x_ts.device)

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

            if t == 0:
                self.forward_at_time_step_0(x_t, h_t, c_t, st_memory, hidden_state_diff)
            else:
                self.forward_one_time_step(x_t, h_t, c_t, st_memory, hidden_state_diff, cell_state_diff)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            if t >= self.seq_len - 1:
                predictions.append(x_gen)

        dec_out = torch.stack(predictions, dim=1)
        
        # patch recovery
        dec_out = rearrange(dec_out, 'b t (c p1 p2) h w -> b t (p1 h) (p2 w) c', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        return dec_out, None