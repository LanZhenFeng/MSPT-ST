import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from numpy import *
from numpy.linalg import *
from scipy.special import factorial
from functools import reduce

from timm.models.swin_transformer import SwinTransformerBlock, window_reverse, PatchEmbed, PatchMerging, window_partition
from timm.layers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.vision_transformer import Block as ViTBlock


# ==============================
#            ConvLSTM       
# ==============================
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


# ==============================
#             PredRNN      
# ==============================
class SpatioTemporalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


# ==============================
#            PredRNN++      
# ==============================
class CausalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(CausalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(m_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c2m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)

        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m = self.conv_om(m_new)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size,
                 stride, layer_norm, initializer=0.001):
        super(GHU, self).__init__()

        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm

        if layer_norm:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )


        if initializer != -1:
            self.initializer = initializer
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.uniform_(m.weight, -self.initializer, self.initializer)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, z):
        if z is None:
            z = self._init_state(x)
        z_concat = self.z_concat(z)
        x_concat = self.x_concat(x)

        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new


# ==============================
#            PredRNNv2      
# ==============================
class SpatioTemporalLSTMCellv2(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellv2, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


# ===============================
#               MIM      
# ==============================
class MIMBlock(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMBlock, self).__init__()

        self.convlstm_c = None
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden*2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
        else:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def MIMS(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1,2,1,1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ +  i_x
            f_ = f_ + f_x
            g_ = g_ +  g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        h = self._init_state(x) if h is None else h
        c = self._init_state(x) if c is None else c
        m = self._init_state(x) if m is None else m
        diff_h = self._init_state(x) if diff_h is None else diff_h

        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c \
            if self.convlstm_c is None else self.convlstm_c.detach())

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        new_h = o * torch.tanh(self.conv_last(cell))

        return new_h, new_c, new_m


class MIMN(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMN, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden*2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
        else:
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1,2,1,1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


# ============================== 
#              MAU      
# ==============================
class MAUCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        # self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.padding = filter_size // 2
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend
        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s

        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        return T_new, S_new


# ==============================
#             E3DLSTM      
# ==============================
class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length,
                 height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = (0, filter_size[1] // 2, filter_size[2] // 2) 
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, window_length, height, width])
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, _, width, height = in_query.shape
        query = in_query.reshape(batch, -1, num_channels)
        keys = in_keys.reshape(batch, -1, num_channels)
        values = in_values.reshape(batch, -1, num_channels)
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = \
            torch.split(x_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)

        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)

        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory


# ==============================
#             PhyDNet      
# ==============================
class PhyCell_Cell(nn.Module):

    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim,
                                             kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm(7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim,
                                             kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3,3),
                                  padding=(1,1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):

    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size)  # init Hidden at each forward start
        for j,cell in enumerate(self.cell_list):
            if j==0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        return self.H, self.H
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append(torch.zeros(
                batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class PhyD_ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(PhyD_ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PhyD_ConvLSTM(nn.Module):

    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(PhyD_ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(PhyD_ConvLSTM_Cell(input_shape=self.input_shape,
                                                input_dim=cur_input_dim,
                                                hidden_dim=self.hidden_dims[i],
                                                kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append(torch.zeros(
                batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(torch.zeros(
                batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3),
                          stride=stride, padding=1),
                nn.GroupNorm(16, nout),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if stride==2:
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3,3),
                                   stride=stride, padding=1, output_padding=output_padding),
                nn.GroupNorm(16, nout),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(encoder_E, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2) # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=stride_2) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(decoder_D, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        output_padding = 1 if stride_2==2 else 0
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3,3),
                                       stride=stride_2, padding=1,
                                       output_padding=output_padding)  #(nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class PhyD_EncoderRNN(torch.nn.Module):

    def __init__(self, phycell, convcell, in_channel=1, patch_size=4):
        super(PhyD_EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(nc=in_channel, patch_size=patch_size) # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific() # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific()
        self.decoder_Dp = decoder_specific() # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D(nc=in_channel, patch_size=patch_size) # general decoder 32x32x32 -> 64x64x1

        self.phycell = phycell
        self.convcell = convcell

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input) # general encoder 64x64x1 -> 32x32x32
    
        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)    # [2, 64, 16, 16]

        hidden1, output1 = self.phycell(input_phys, first_timestep) #[[2, 64, 16, 16]]
        hidden2, output2 = self.convcell(input_conv, first_timestep) #[[2, 64, 16, 16]]

        decoded_Dp = self.decoder_Dp(output1[-1]) # [2, 64, 16, 16]
        decoded_Dr = self.decoder_Dr(output2[-1]) # [2, 64, 16, 16]
        
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp)) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr # [2, 64, 16, 16]
        output_image = torch.sigmoid(self.decoder_D(concat)) # [2, 11, 64, 64]
        return out_phys, hidden1, output_image, out_phys, out_conv


def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous()
    x = x.view(sizex)
    return x


def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    x = x.permute(list(range(1,k+1))+[0,])
    for i in range(k):
        x = tensordot(x, mats[i], dim=[0,0])
    x = x.contiguous()
    x = x.view(sizex)
    return x


class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(zeros((l,l)))
            for i in range(l):
                M[-1][i] = ((arange(l)-(l-1)//2)**i)/factorial(i)
            invM.append(inv(M[-1]))
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))
    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        return self._size
    def dim(self):
        return self._dim
    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """
    def __init__(self, shape):
        super(M2K, self).__init__(shape)
    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m


class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x,y:x*y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1,N]).double()
    b = b.view([N,-1]).double()
    c = a@b
    return c.view(sizea0+sizeb1)


# ==============================
#            SwinLSTM      
# ==============================
class SwinLSTMCell(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        """
        Args:
        flag:  0 UpSample   1 DownSample  2 STconvert
        """
        super(SwinLSTMCell, self).__init__()

        self.STBs = nn.ModuleList(
            STB(i, dim=dim, input_resolution=input_resolution, depth=depth, 
                num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, flag=flag)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        """
        Args:
        xt: input for t period 
        hidden_states: [hx, cx] hidden_states for t-1 period
        """
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)

        else:
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.STBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)
                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)

        cell = torch.tanh(o_t)

        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)

class STB(SwinTransformerBlock):
    def __init__(self, index, dim, input_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        if flag == 0:
            drop_path = drop_path[depth - index - 1]
        elif flag == 1:
            drop_path = drop_path[index]
        elif flag == 2:
            drop_path = drop_path
        super(STB, self).__init__(dim=dim, input_resolution=input_resolution, 
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=0 if (index % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,                                   
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path,
                                  norm_layer=norm_layer)
        self.red = nn.Linear(2 * dim, dim)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        
class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)

        return x
       
class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, flag=0):
        super(UpSample, self).__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)

            layer = SwinLSTMCell(dim=dimension, input_resolution=(resolution1, resolution2),
                                 depth=depths_upsample[(self.num_layers - 1 - i_layer)],
                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                               sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                 norm_layer=norm_layer, flag=flag)

            self.layers.append(layer)
            self.upsample.append(upsample)

    # def forword_core(self, x, y):
    #     hidden_states_up = []

    #     for index, layer in enumerate(self.layers):
    #         x, hidden_state = layer(x, y[index])
    #         x = self.upsample[index](x)
    #         hidden_states_up.append(hidden_state)

    #     return hidden_states_up, x

    def forward(self, x, y):
        hidden_states_up = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        # hidden_states_up, x = self.forword_core(x, y)

        x = torch.sigmoid(self.Unembed(x))

        return hidden_states_up, x

class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, flag=1):
        super(DownSample, self).__init__()

        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))

            layer = SwinLSTMCell(dim=int(embed_dim * 2 ** i_layer),
                                 input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                                 depth=depths_downsample[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                 norm_layer=norm_layer, flag=flag)

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):

        x = self.patch_embed(x)

        hidden_states_down = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)

        return hidden_states_down, x

class STconvert(nn.Module): 
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, num_heads, 
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, flag=2):
        super(STconvert, self).__init__()
        
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim, 
                                      norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size

        self.patch_inflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim,
                                            input_resolution=patches_resolution)

        self.layer = SwinLSTMCell(dim=embed_dim, 
                                  input_resolution=(patches_resolution[0], patches_resolution[1]), 
                                  depth=depths, num_heads=num_heads,
                                  window_size=window_size, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=drop_path_rate, norm_layer=norm_layer,
                                  flag=flag)
    def forward(self, x, h=None):
        x = self.patch_embed(x)

        x, hidden_state = self.layer(x, h)

        x = torch.sigmoid(self.patch_inflated(x))
        
        return x, hidden_state


# ==============================
#            CrevNet      
# ==============================
def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :, :].contiguous()
    x2 = x[:, n:, :, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class psi(nn.Module):

    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / 4)
        s_width = int(d_width * 2)
        s_height = int(d_height * 2)
        t_1 = output.contiguous().view(batch_size, temp, d_height, d_width, 4, s_depth)
        spl = t_1.split(2, 4)
        stack = [t_t.contiguous().view(batch_size, temp, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).transpose(1, 2).permute(
            0, 1, 3, 2, 4, 5).contiguous().view(batch_size, temp, s_height, s_width, s_depth)
        output = output.permute(0, 4, 1, 2, 3)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 4, 1)
        (batch_size, temp, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size,temp, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 2)
        output = output.permute(0, 4, 1, 3, 2)
        return output.contiguous()


class irevnet_block(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=2):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.stride = stride
        self.psi = psi(stride)
        layers = []
        if not first:
            layers.append(nn.BatchNorm3d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        if self.stride ==2:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=(1,2,2), padding=1, bias=False))
        else:
            layers.append(nn.Conv3d(in_ch // 2, ch, kernel_size=3,
                                    stride=self.stride, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, ch,
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm3d(ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(ch, out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x


class autoencoder(nn.Module):

    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=2):
        super(autoencoder, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)

        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict = True):
        if is_predict:
            n = self.in_ch // 2
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            out = (x[:, :n, :, :, :], x[:, n:, :, :, :])

            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            x = self.init_psi.inverse(out)
        return x


class STConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, memo_size):
        super(STConvLSTMCell,self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memo_size = memo_size
        self.in_gate = nn.Conv3d(
            input_size + hidden_size + hidden_size , hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate = nn.Conv3d(
            input_size + hidden_size + hidden_size , hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate = nn.Conv3d(
            input_size + hidden_size + hidden_size, hidden_size , self.KERNEL_SIZE, padding=self.PADDING)

        self.in_gate1 = nn.Conv3d(
            input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate1 = nn.Conv3d(
            input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate1 = nn.Conv3d(
            input_size + memo_size + hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w1 = nn.Conv3d(hidden_size + memo_size, hidden_size, 1)
        self.out_gate = nn.Conv3d(
            input_size + hidden_size +hidden_size+memo_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)


    def forward(self, input, prev_state):
        input_,prev_memo = input
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden, prev_cell), 1)

        in_gate = F.sigmoid(self.in_gate(stacked_inputs))
        remember_gate = F.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = F.tanh(self.cell_gate(stacked_inputs))

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo, cell), 1)

        in_gate1 = F.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = F.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = F.tanh(self.cell_gate1(stacked_inputs1))

        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = F.sigmoid(self.out_gate(torch.cat((input_, prev_hidden, cell, memo), 1)))
        hidden = out_gate * F.tanh(self.w1(torch.cat((cell, memo), 1)))

        return (hidden, cell),memo


class zig_rev_predictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, batch_size, temp =3, w =8,h = 8):
        super(zig_rev_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.temp = temp
        self.w = w
        self.h = h

        self.convlstm = nn.ModuleList(
                [STConvLSTMCell(input_size, hidden_size, hidden_size) if i == 0 else \
                    STConvLSTMCell(hidden_size, hidden_size, hidden_size) for i in
                 range(self.n_layers)])

        self.att = nn.ModuleList([nn.Sequential(nn.Conv3d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                nn.Sigmoid()
                                                ) for i in range(self.n_layers)])

        self.hidden = self.init_hidden()
        self.prev_hidden = self.hidden

    def init_hidden(self):
        hidden = []

        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size,self.temp,self.w,self.h).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size,self.temp,self.w,self.h).cuda())))
        return hidden

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
        for i in range(self.n_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.att[i](self.hidden[i][0])
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1,x2),memo


# ==============================
#            DFVFN      
# ==============================
backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)
        # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border',
                                           align_corners=True)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class Routing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Routing, self).__init__()
        self.routing = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.routing(x)


class MVFB(nn.Module):
    def __init__(self, in_planes, num_feature):
        super(MVFB, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature//2, 3, 2, 1),
            conv(num_feature//2, num_feature, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature//4)

        self.conv1 = nn.Sequential(
            conv(in_planes, 8, 3, 2, 1),
        )
        self.convblock1 = nn.Sequential(
            conv(8, 8),
        )
        self.lastconv = nn.ConvTranspose2d(num_feature//4 + 8, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.interpolate(x2, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        x3 = self.conv1(torch.cat((x0,flow0), 1))
        x4 = self.convblock1(x3)
        tmp = self.lastconv(torch.cat((x2, x4), dim=1))
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask


# ------------------------------
#           SimVP
# ------------------------------
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HorBlock(nn.Module):
    """ HorNet block """

    def __init__(self, dim, order=4, mlp_ratio=4, drop_path=0., init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim, order)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        gamma1 = self.gamma1.view(C, 1, 1)
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation in MogaNet."""

    def __init__(self, embed_dims, mlp_hidden_dims, kernel_size=3, act_layer=nn.GELU, ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_hidden_dims = mlp_hidden_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=self.mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=self.mlp_hidden_dims, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True, groups=self.mlp_hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dims, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(
            1e-5 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel in MogaNet."""

    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims, stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation in MogaNet."""

    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_shortcut=True):
        super(MultiOrderGatedAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.attn_shortcut = attn_shortcut
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        # decompose
        self.sigma = nn.Parameter(1e-5 * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma * (x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., drop=0., drop_path=0.,
                 init_value=1e-5, act_layer=nn.GELU, norm_layer=GroupNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_value=1e-6, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # layer scale
        self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_1', 'gamma_2'}

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MixMlp(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class VANBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., init_value=1e-2, act_layer=nn.GELU, attn_shortcut=True):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, attn_shortcut=attn_shortcut)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class ConvMixerSubBlock(nn.Module):
    """A block of ConvMixer."""

    def __init__(self, dim, kernel_size=9, activation=nn.GELU):
        super().__init__()
        # spatial mixing
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        # channel mixing
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        x = x + self.norm_1(self.act_1(self.conv_dw(x)))
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, ls_init_value=1e-6, conv_mlp=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma'}

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) * self.mlp(self.norm(self.conv_dw(x))))
        return x


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4., drop_path=0.1, init_value=1e-6):
        super().__init__(dim, mlp_ratio=mlp_ratio, drop_path=drop_path, init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MLPMixerSubBlock(MixerBlock):
    """A block of MLP-Mixer."""

    def __init__(self, dim, input_resolution=None, mlp_ratio=4., drop=0., drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(dim, seq_len=seq_len,
                         mlp_ratio=(0.5, mlp_ratio), drop_path=drop_path, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, mlp_ratio=4., drop_rate=0., drop_path_rate=0., init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        # spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation=attn_dw_dilation, attn_channel_split=attn_channel_split)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path,
                         drop=drop, init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, drop=drop, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


def UniformerSubBlock(embed_dims, mlp_ratio=4., drop=0., drop_path=0.,
                      init_value=1e-6, block_type='Conv'):
    """Build a block of Uniformer."""

    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        return SABlock(dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                       drop=drop, drop_path=drop_path, init_value=init_value)


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                         init_value=init_value, act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class ViTSubBlock(ViTBlock):
    """A block of Vision Transformer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim=dim, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                         drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    

class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    

class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u


class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                 drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        
        self.attn = TemporalAttention(dim, kernel_size)