import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.OpenSTL_Modules import DownSample, UpSample


class SwinLSTM_D_Model(nn.Module):
    r"""SwinLSTM 
    Implementation of `SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin
    Transformer and LSTM <http://arxiv.org/abs/2308.09891>`_.

    """

    def __init__(self, depths_downsample, depths_upsample, num_heads, configs, **kwargs):
        super(SwinLSTM_D_Model, self).__init__()
        T, C, H, W = configs.in_shape
        assert H == W, 'Only support H = W for image input'
        self.configs = configs
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample
        self.Downsample = DownSample(img_size=H, patch_size=configs.patch_size, in_chans=C,
                                     embed_dim=configs.embed_dim, depths_downsample=depths_downsample,
                                     num_heads=num_heads, window_size=configs.window_size)

        self.Upsample = UpSample(img_size=H, patch_size=configs.patch_size, in_chans=C,
                                     embed_dim=configs.embed_dim, depths_upsample=depths_upsample,
                                     num_heads=num_heads, window_size=configs.window_size)
        self.MSE_criterion = nn.MSELoss()

    def forward(self, frames_tensor, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        T, C, H, W = self.configs.in_shape
        total_T = frames_tensor.shape[1]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()

        input_frames = frames[:, :T]
        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)
        next_frames = []
        last_frame = input_frames[:, -1]
        
        for i in range(T - 1):
            states_down, x = self.Downsample(input_frames[:, i], states_down) 
            states_up, output = self.Upsample(x, states_up)
            next_frames.append(output)
        for i in range(total_T - T):
            states_down, x = self.Downsample(last_frame, states_down) 
            states_up, output = self.Upsample(x, states_up)
            next_frames.append(output)
            last_frame = output
 

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss


class Model(nn.Module):
    r"""SwinLSTM 

    Implementation of `SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin
    Transformer and LSTM <http://arxiv.org/abs/2308.09891>`_.

    """

    def __init__(self, configs, depths_downsample=[2, 6], depths_upsample=[6, 2], num_heads=[4, 8], patch_size=2, window_size=4):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size 

        # curriculum learning strategy
        self.cls = configs.curriculum_learning_strategy # RSS, SS or Standard
        curriculum_learning_strategies = ['rss', 'ss', 's']
        assert self.cls in curriculum_learning_strategies, "curriculum_learning_strategy must be one of ['rss', 'ss', 's']"

        self.Downsample = DownSample(img_size=[configs.height, configs.width], patch_size=patch_size, in_chans=configs.enc_in,
                                     embed_dim=configs.d_model, depths_downsample=depths_downsample,
                                     num_heads=num_heads, window_size=window_size)

        self.Upsample = UpSample(img_size=[configs.height, configs.width], patch_size=patch_size, in_chans=configs.enc_in,
                                     embed_dim=configs.d_model, depths_upsample=depths_upsample,
                                     num_heads=num_heads, window_size=window_size)

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

        x_ts = rearrange(x_enc, 'b t h w c -> b t c h w')

        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)
        
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

            states_down, x = self.Downsample(x_t, states_down) 
            states_up, x_gen = self.Upsample(x, states_up)
            if t >= self.seq_len:
                predictions.append(x_gen)

        dec_out = torch.stack(predictions, dim=1)
        
        dec_out = rearrange(dec_out, 'b t c h w -> b t h w c')
        
        return dec_out, None