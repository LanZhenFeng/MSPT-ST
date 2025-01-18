import math
import torch
import torch.nn as nn

from einops import rearrange

from layers.OpenSTL_Modules import MAUCell


class MAU_Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MAU_Model, self).__init__()
        T, C, H, W = configs.in_shape
        
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = W // configs.patch_size // configs.sr_size
        height = H // configs.patch_size // configs.sr_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)
        
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames[:, 1:])
        else:
            loss = None

        return next_frames, loss


class Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, configs, sr_size=4, tau=5, cell_mode='normal', model_mode='normal', filter_size=5, stride=1, layer_norm=False):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        self.num_hidden = configs.d_model
        self.sr_size = sr_size
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm

        self.tau = tau
        self.cell_mode = cell_mode
        self.model_mode = model_mode
        self.states = ['recall', 'normal']
        if not self.model_mode in self.states:
            raise AssertionError

        # curriculum learning strategy
        self.cls = configs.curriculum_learning_strategy # RSS, SS or Standard
        curriculum_learning_strategies = ['rss', 'ss', 's']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 's']"

        cell_list = []
        self.d_patch = configs.patch_size * configs.patch_size * configs.enc_in
        height = configs.height // configs.patch_size // self.sr_size
        width = configs.width // configs.patch_size // self.sr_size
        for i in range(self.num_layers):
            in_channel = self.d_patch if i == 0 else self.num_hidden
            cell_list.append(MAUCell(in_channel, self.num_hidden, height, width, self.filter_size, self.stride, self.layer_norm, self.tau, self.cell_mode))
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(self.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1), module=nn.Conv2d(in_channels=self.d_patch, out_channels=self.num_hidden, stride=1, padding=0, kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1), module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i), module=nn.Conv2d(in_channels=self.num_hidden, out_channels=self.num_hidden, stride=(2, 2), padding=(1, 1), kernel_size=(3, 3)))
            encoder.add_module(name='encoder_t_relu{0}'.format(i), module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i), module=nn.ConvTranspose2d(in_channels=self.num_hidden, out_channels=self.num_hidden, stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1)))
            decoder.add_module(name='c_decoder_relu{0}'.format(i), module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1), module=nn.ConvTranspose2d(in_channels=self.num_hidden, out_channels=self.num_hidden, stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1)))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.conv_last = nn.Conv2d(self.num_hidden, self.d_patch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward_one_time_step(self, x_t, T_t, T_pre, S_pre):
        frames_feature = x_t
        frames_feature_encoded = []
        for i in range(len(self.encoders)):
            frames_feature = self.encoders[i](frames_feature)
            frames_feature_encoded.append(frames_feature)
        S_t = frames_feature
        for i in range(self.num_layers):
            t_att = T_pre[i][-self.tau:]
            t_att = torch.stack(t_att, dim=0)
            s_att = S_pre[i][-self.tau:]
            s_att = torch.stack(s_att, dim=0)
            S_pre[i].append(S_t)
            T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
            T_pre[i].append(T_t[i])
        for i in range(len(self.decoders)):
            out = self.decoders[i](out)
            if self.model_mode == 'recall':
                out = out + frames_feature_encoded[-2 - i]
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        if kwargs.get('mask_true', None) is not None:
            mask_true = kwargs['mask_true']
        else:
            if self.cls in ['rss', 'ss']:
                raise ValueError("mask_true is required for RSS and SS")
            mask_true = torch.zeros(x_enc.shape[0], self.pred_len, x_enc.shape[2], x_enc.shape[3], x_enc.shape[4], device=x_enc.device)
        
        if kwargs.get('batch_y', None) is not None:
            batch_y = kwargs['batch_y']
            x_enc = torch.cat([x_enc, batch_y[:, -self.seq_len:]], dim=1)
        else:
            if self.cls in ['rss', 'ss']:
                raise ValueError("batch_y is required for RSS and SS")

        # patching
        x_ts = rearrange(x_enc, 'b t (p1 h) (p2 w) c -> b t (c p1 p2) h w', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        B, T, D, H, W = x_ts.shape

        T_t = []
        T_pre = []
        S_pre = []
        for i in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            for i in range(self.tau):
                tmp_t.append(torch.zeros([B, self.num_hidden, H // self.sr_size, W // self.sr_size], device=x_ts.device))
                tmp_s.append(torch.zeros([B, self.num_hidden, H // self.sr_size, W // self.sr_size], device=x_ts.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)
            T_t.append(torch.zeros([B, self.num_hidden, H // self.sr_size, W// self.sr_size], device=x_ts.device))

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

            out = self.forward_one_time_step(x_t, T_t, T_pre, S_pre)
            x_gen = self.conv_last(out)
            if t >= self.seq_len:
                predictions.append(x_gen)
        
        dec_out = torch.stack(predictions, dim=1)

        # patch recovery
        dec_out = rearrange(dec_out, 'b t (c p1 p2) h w -> b t (p1 h) (p2 w) c', p1=self.configs.patch_size, p2=self.configs.patch_size)
        
        return dec_out, None