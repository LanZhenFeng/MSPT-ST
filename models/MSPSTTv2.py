import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil, log
from einops import rearrange
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.models.vision_transformer import Attention as VariableAttention
from timm.layers import Mlp, LayerNorm2d, use_fused_attn, to_2tuple

# add layers to the system path ../../layers
import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上上层目录
parent_dir = os.path.dirname(current_dir)
# 将上上层目录添加到 sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from layers.Embed import PositionalEmbedding, PositionalEmbedding2D

def dispatch(inp, gates):
    # sort experts
    _, index_sorted_experts = torch.nonzero(gates).sort(0)
    # get according batch index for each expert
    _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
    _part_sizes = (gates > 0).sum(0).tolist()
    # assigns samples to experts whose gate is nonzero
    # expand according to batch index so we can just split by _part_sizes
    inp_exp = inp[_batch_index].squeeze(1)
    return torch.split(inp_exp, _part_sizes, dim=0)


def combine(expert_out, gates, position_wise=False, multiply_by_gates=True):
    # sort experts
    sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
    _, _expert_index = sorted_experts.split(1, dim=1)
    # get according batch index for each expert
    _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
    gates_exp = gates[_batch_index.flatten()]
    _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)
    # apply exp to expert outputs, so we are not longer in log space
    stitched = torch.cat(expert_out, 0).exp()
    if multiply_by_gates:
        if position_wise:
            stitched = torch.einsum("bctd,bk -> bctd", stitched, _nonzero_gates)
        else:
            stitched = torch.einsum("bcthwd,bk -> bcthwd", stitched, _nonzero_gates)
    if position_wise:
        zeros = torch.zeros(gates.size(0),
                            expert_out[-1].size(1),
                            expert_out[-1].size(2),
                            expert_out[-1].size(3),
                            requires_grad=True,
                            device=stitched.device)
    else:
        zeros = torch.zeros(gates.size(0),
                            expert_out[-1].size(1),
                            expert_out[-1].size(2),
                            expert_out[-1].size(3),
                            expert_out[-1].size(4),
                            expert_out[-1].size(5),
                            requires_grad=True,
                            device=stitched.device)
    # combine samples that have been processed by the same k experts
    combined = zeros.index_add(0, _batch_index, stitched.float())
    # add eps to all zero values in order to avoid nans when going back to log space
    combined[combined == 0] = np.finfo(float).eps
    # back to log space
    return combined.log()


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)


class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop=0., is_causal=False, output_attention=False):
        super().__init__()
        assert d_model % n_heads == 0, 'dim should be divisible by num_heads'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_drop = nn.Dropout(attn_drop)
        self.is_causal = is_causal
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, H, L, E = queries.shape
        _, H, S, D = values.shape

        if self.fused_attn and not self.output_attention:
            return F.scaled_dot_product_attention(
                queries, keys, values, attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=self.is_causal,
            ), None
        else:
            attn_bias = torch.zeros(L, S, dtype=queries.dtype, device=queries.device)
            if self.is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(queries.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias = attn_mask + attn_bias

            queries = queries * self.scale
            attn_weight = queries @ keys.transpose(-2, -1)
            attn_weight = attn_weight.softmax(dim=-1)
            attn_weight = self.attn_drop(attn_weight)
            if self.output_attention:
                return attn_weight @ values, attn_weight
            else:
                return attn_weight @ values, None


class VariableAttentionLayer(nn.Module):
    def __init__(self, attention, in_chans, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0.):
        super(VariableAttentionLayer, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.inner_attention = attention
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.learned_pe = learned_pe
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans, d_model)) if learned_pe else PositionalEmbedding(d_model)
        self.pos_drop = nn.Dropout(pos_drop)


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape
        x = rearrange(x, 'b t c h w d -> (b t h w) c d')

        # positional embedding
        if self.learned_pe:
            x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x + self.pos_embed(x))

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'bthw n (qkv h d) -> qkv bthw h n d', qkv=3, h=self.n_heads)
        queries, keys, values = qkv.unbind(0)
        queries, keys = self.q_norm(queries), self.k_norm(keys)

        x, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        x = rearrange(x, 'bthw h n d -> bthw n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(b t h w) c d -> b t c h w d', b=B, t=T, h=H, w=W)
        return x, attn


class FourierLayer(nn.Module):
    def __init__(self, seq_len, top_k, d_model, in_chans, img_size, fuse_drop=0., position_wise=False, individual=False):
        super(FourierLayer, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_freqs = seq_len // 2
        self.position_wise = position_wise
        self.height, self.width = img_size if not position_wise else (1, 1)
        
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)
        # # get the Highest Common Factor
        # hcf = np.gcd(self.height, self.width).astype(int)
        # self.num_embed_layers = np.log2(hcf).astype(int)
        # self.embed_layers = nn.ModuleList()
        # for i in range(self.num_embed_layers):
        #     input_size = d_model * 2**i
        #     output_size = d_model * 2**(i+1)
        #     self.embed_layers.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=2, stride=2), 
        #             nn.GELU(),
        #             nn.BatchNorm2d(output_size),
        #         )
        #     )

        # in_chans = in_chans if individual else 1
        # input_size = in_chans * d_model if position_wise else d_model * in_chans * 2**self.num_embed_layers * self.height//2**self.num_embed_layers * self.width//2**self.num_embed_layers
        # self.fuse_proj = nn.Linear(input_size, 512)
        # self.fuse_drop = nn.Dropout(fuse_drop)

        # if not position_wise:
        #     self.squeeze_layer = nn.Sequential(
        #         nn.Conv2d(in_channels=d_model, out_channels=d_model*4, kernel_size=4, stride=2, padding=1),
        #         nn.GroupNorm(16, d_model*4),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.AdaptiveAvgPool2d((1, 1)),
        #         nn.Conv2d(in_channels=d_model*4, out_channels=d_model, kernel_size=1, stride=1),
        #         nn.Flatten()
        #     )
        
        self.in_proj = nn.Linear(in_chans * d_model, 512) if individual else nn.Linear(d_model, 512)

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

    def get_segment_sizes(self, seq_len):
        # get the period list, first element is inf
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
        # segment_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        # print(f"Segment sizes: {segment_sizes}")
        return segment_sizes

    def forward(self, x, training, noise_epsilon=1e-2):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape

        # # embed & fuse
        # if self.position_wise:
        #     x = rearrange(x, 'b t c h w d -> (b h w) t (c d)')
        # else:
        #     x = rearrange(x, 'b t c h w d -> (b t c) d h w')
        #     for embed_layer in self.embed_layers:
        #         x = embed_layer(x)
        #     x = rearrange(x, '(b t c) d h w -> b t (c h w d)', c=C, t=T)
        # x = self.fuse_proj(x)
        # x = self.fuse_drop(x)
        # if self.position_wise:
        #     x = rearrange(x, 'b t c h w d -> (b h w) t (c d)')
        # else:
        #     x = rearrange(x, 'b t c h w d -> (b t c) d h w')
        #     x = self.squeeze_layer(x) # -> [BTC, D]
        #     x = rearrange(x, '(b t c) d -> b t (c d)', t=T, c=C)
        # x = self.in_proj(x)
        if self.position_wise:
            x = rearrange(x, 'b t c h w d -> (b h w) t (c d)')
        else:
            x = rearrange(x, 'b t c h w d -> b t (h w) (c d)')
            x = x.mean(-2) 
        x = self.in_proj(x)

        # fft
        x = rearrange(x, 'b t d -> b d t')
        xf = torch.fft.rfft(x, dim=-1, norm='ortho')[:, :, 1:]  # [B, D, T//2]
        amp = torch.abs(xf) # [B, D, T//2]

        clean_logits = amp @ self.w_gate # [B, D, Ps]
        if training:
            raw_noise_stddev = amp @ self.w_noise # [B, D, Ps]
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            logits = noisy_logits  # [B, D, Ps]
        else:
            logits = clean_logits  # [B, D, Ps]

        weights = logits.mean(1)  # [B, Ps]
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)  # [B, top_k], [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(weights)  # [B, Ps]
        gates = zeros.scatter_(-1, top_indices, top_weights)  # [B, Ps]

        return gates  # [B, Ps]


class PeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, segment_size, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0., position_wise=False, individual=False):
        super(PeriodicAttentionLayer, self).__init__()
        assert segment_size*d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.seq_len = seq_len
        self.segment_size = segment_size
        self.d_model = d_model * segment_size
        self.inner_attention = attention
        self.n_heads = n_heads
        self.head_dim = self.d_model // n_heads
        self.qkv = nn.Linear(self.d_model, self.d_model * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(self.d_model, self.d_model, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.position_wise = position_wise
        self.individual = individual

        self.learned_pe = learned_pe
        num_tokens = seq_len // segment_size if seq_len % segment_size == 0 else seq_len // segment_size + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.d_model)) if learned_pe else PositionalEmbedding(self.d_model)
        self.pos_drop = nn.Dropout(pos_drop)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D] or [BHW, T, C, D]
        if x.shape[0] == 0:
            return x, None

        if self.position_wise:
            B, T, C, D = x.shape
            x = rearrange(x, 'b t c d -> (b c) t d')
        else:
            B, T, C, H, W, D = x.shape
            x = rearrange(x, 'b t c h w d -> (b c h w) t d')

        # segment
        padding_len = T - T % self.segment_size if T % self.segment_size != 0 else 0
        x = F.pad(x, (0, 0, 0, padding_len), mode='replicate')
        x = x.unfold(1, self.segment_size, self.segment_size).contiguous()
        x = rearrange(x, 'b n d p -> b n (p d)')

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.n_heads)
        queries, keys, values = qkv.unbind(0)
        queries, keys = self.q_norm(queries), self.k_norm(keys)

        x, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'b n (p d) -> b (n p) d', p=self.segment_size)[:,:T]
        if self.position_wise:
            x = rearrange(x, '(b c) t d -> b t c d', c=C)
        else:
            x = rearrange(x, '(b c h w) t d -> b t c h w d', c=C, h=H, w=W)

        return x, attn


class MultiScalePeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, gate_layer, seq_len, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0., position_wise=False, individual=False):
        super(MultiScalePeriodicAttentionLayer, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.gate_layer = gate_layer
        self.segment_sizes = gate_layer.segment_sizes

        self.attention_layers = nn.ModuleList()
        for segment_size in self.segment_sizes:
            self.attention_layers.append(
                PeriodicAttentionLayer(
                    attention,
                    seq_len,
                    segment_size,
                    d_model,
                    n_heads,
                    learned_pe=learned_pe,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop,
                    pos_drop=pos_drop,
                    position_wise=position_wise,
                    individual=individual
                )
            )

        self.position_wise = position_wise

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape

        # gating
        gates = self.gate_layer(x, training=self.training) # [B, Ps]

        # re-arrange
        if self.position_wise:
            x = rearrange(x, 'b t c h w d -> (b h w) t c d')

        # dispatch
        x = dispatch(x, gates) # Ps * [B, T, C, H, W, D] or Ps * [BHW, T, C, D]
        
        # multi-branch attention
        xs = []
        attns = []
        for i, attention_layer in enumerate(self.attention_layers):
            xi, attn = attention_layer(x[i], attn_mask=attn_mask, tau=tau, delta=delta)
            xs.append(xi)
            attns.append(attn)

        # combine
        x = combine(xs, gates, position_wise=self.position_wise)

        # re-arrange
        if self.position_wise:
            x = rearrange(x, '(b h w) t c d -> b t c h w d', h=H, w=W)

        return x, attns

class MyWindowAttention(WindowAttention):
    def __init__(self, d_model, n_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0., qk_norm=False, output_attention=False):
        super(MyWindowAttention, self).__init__(d_model, n_heads, head_dim, window_size, qkv_bias, attn_drop, proj_drop)
        head_dim = head_dim or d_model // n_heads
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.output_attention = output_attention

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)
        queries, keys, values = qkv.unbind(0)
        queries, keys = self.q_norm(queries), self.k_norm(keys)

        if self.fused_attn and not self.output_attention:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=attn_mask, 
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            queries = queries * self.scale
            attn_weight = queries @ keys.transpose(-2, -1)
            attn_weight = attn_weight + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn_weight = attn_weight.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_weight = attn_weight.view(-1, self.num_heads, N, N)
            attn_weight = self.softmax(attn_weight)
            attn_weight = self.attn_drop(attn_weight)
            x = attn_weight @ values

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.output_attention:
            return x, attn_weight
        else:
            return x, None


class WindowAttentionLayer(nn.Module):
    def __init__(self, attention, input_resolution, window_size=7, shift_size=0, always_partition=False, dynamic_mask=False):
        super(WindowAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.input_resolution = input_resolution
        self.target_shift_size = to_2tuple(shift_size)
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def get_attn_mask(self, x=None):
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is not None:
                H, W = x.shape[1], x.shape[2]
                device = x.device
                dtype = x.dtype
            else:
                H, W = self.input_resolution
                device = None
                dtype = None
            H = ceil(H / self.window_size[0]) * self.window_size[0]
            W = ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1), dtype=dtype, device=device)  # 1 H W 1
            cnt = 0
            for h in ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None),):
                for w in ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None),):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def _calc_window_shift(self, target_window_size, target_shift_size=None):
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was previously non-zero
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(self, feat_size, window_size, always_partition=None):
        """
        Args:
            feat_size: New input resolution
            window_size: New window size
            always_partition: Change always_partition attribute if not None
        """
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.inner_attention.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def forward(self, x):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape
        
        # re-arrange
        x = rearrange(x, 'b t c h w d -> (b t c) h w d')

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # pad for resolution not divisible by window size
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, D
        x_windows = x_windows.view(-1, self.window_area, D)  # nW*B, window_size*window_size, D

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows, attn = self.inner_attention(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, D

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], D)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' D
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x

        # re-arrange
        x = rearrange(x, '(b t c) h w d -> b t c h w d', b=B, c=C, t=T)

        return x, attn


class MSPSTTEncoderLayer(nn.Module):
    # r"""Parallel Spatial-Temporal Attention Block with Shared Variable Attention"""
    def __init__(self, attention_v, attention_t, attention_s, d_model, d_ff=None, dropout=0., activation="relu", pre_norm=False, is_parallel=True):
        super(MSPSTTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_v = attention_v
        self.attention_t = attention_t
        self.attention_s = attention_s
        activation = nn.ReLU if activation == "relu" else nn.GELU
        self.mlp_t = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.mlp_s = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.fusion_st = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.is_parallel = is_parallel

        self.fusion_st = Mlp(in_features=d_model*2, hidden_features=d_ff, out_features=d_model, act_layer=activation, drop=dropout)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D]
        # B, T, C, H, W, D = x.shape
        # variable attention
        if self.attention_v is not None:
            res = x
            if self.pre_norm:
                x = self.norm1(x)
            x, attn_v = self.attention_v(x, attn_mask=attn_mask, tau=tau, delta=delta)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm1(x)
        else:
            attn_v = None

        if self.is_parallel:
            ## parallel spatial attention and temporal attention
            res_t = res_s = res = x
            # temporal attention
            if self.pre_norm:
                x_t = self.norm2(x)
            else:
                x_t = x
            x_t, attn_t = self.attention_t(x_t, attn_mask=attn_mask, tau=tau, delta=delta)
            x_t = res_t + self.dropout(x_t)
            if not self.pre_norm:
                x_t = self.norm2(x_t)

            # temporal mlp
            res_t = x_t
            if self.pre_norm:
                x_t = self.norm3(x_t)
            x_t = self.mlp_t(x_t)
            x_t = res_t + self.dropout(x_t)
            if not self.pre_norm:
                x_t = self.norm3(x_t)

            # spatial attention
            if self.pre_norm:
                x_s = self.norm3(x)
            else:
                x_s = x
            x_s, attn_s = self.attention_s(x_s)
            x_s = res_s + self.dropout(x_s)
            if not self.pre_norm:
                x_s = self.norm3(x_s)

            # spatial mlp
            res_s = x_s
            if self.pre_norm:
                x_s = self.norm3(x_s)
            x_s = self.mlp_s(x_s)
            x_s = res_s + self.dropout(x_s)
            if not self.pre_norm:
                x_s = self.norm3(x_s)

            # combine
            x = torch.cat([x_t, x_s], dim=-1)
            # x = self.fusion_st(x)
            if self.pre_norm:
                x = self.norm4(x)
            x = self.fusion_st(x)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm4(x)

        else:
            ## serial spatial attention and temporal attention
            res = x
            # temporal attention
            if self.pre_norm:
                x = self.norm2(x)
            x, attn_t = self.attention_t(x, attn_mask=attn_mask, tau=tau, delta=delta)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm2(x)

            # temporal mlp
            res = x
            if self.pre_norm:
                x = self.norm3(x)
            x = self.mlp_t(x)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm3(x)

            # spatial attention
            res = x
            if self.pre_norm:
                x = self.norm3(x)
            x, attn_s = self.attention_s(x, mask=attn_mask)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm3(x)

            # spatial mlp
            res = x
            if self.pre_norm:
                x = self.norm3(x)
            x = self.mlp_s(x)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm3(x)

        return x, (attn_v, attn_t, attn_s)
    

class MSPSTTEncoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None):
        super(MSPSTTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D]
        attns = []

        for encoder_layer in self.encoder_layers:
            x, attn = encoder_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, individual=False):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.individual = individual
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size) if individual else nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, H, W, C]
        B, T, H, W, C = x.shape
        if self.individual:
            x = rearrange(x, 'b t h w c -> (b t c) () h w')
        else:
            x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.proj(x)
        if self.individual:
            x = rearrange(x, '(b t c) d h w -> b t c h w d', t=T, c=C)
        else:
            x = rearrange(x, '(b t) d h w -> b t () h w d', t=T)
        return x


class PatchRecovery(nn.Module):
    r"""Patch Recovery Module [B, C, T, H, W, D] -> [B, T, H, W, C]"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, individual=False):
        super(PatchRecovery, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.individual = individual
        self.proj = nn.ConvTranspose2d(embed_dim, 1, kernel_size=patch_size, stride=patch_size) if individual else nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape
        x = rearrange(x, 'b t c h w d -> (b t c) d h w')
        x = self.proj(x)
        x = rearrange(x, '(b t c) d h w -> b t h w (c d)', t=T, c=C)
        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, patch_size, in_chans, embed_dim, stride=2, padding=1, output_padding=1, individual=False):
        super(PatchInflated, self).__init__()

        num_layers = np.log2(patch_size).astype(int) - 1
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)

        self.Convs = nn.ModuleList()
        for i in range(num_layers):
            self.Convs.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=stride, padding=padding, output_padding=output_padding),
                nn.GroupNorm(16, embed_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        out_chans = in_chans if individual else 1
        self.Convs.append(nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=(3, 3), stride=stride, padding=padding, output_padding=output_padding))

        self.individual = individual

    def forward(self, x):
        B, T, C, H, W, D = x.shape

        x = rearrange(x, 'b t c h w d -> (b t c) d h w')
        for conv_layer in self.Convs:
            x = conv_layer(x)
        x = rearrange(x, '(b t c) d h w -> b t h w (c d)', t=T, c=C)
        
        return x


class Model(nn.Module):
    r'''
    
    Multi-Scale Periodic Spatio-Temporal Transformer ------ SimVP Architecture
    
    '''
    def __init__(self, configs, window_size=4): 
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.height = configs.height
        self.width = configs.width
        self.img_size = tuple([configs.height, configs.width])
        self.patch_size = configs.patch_size
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])
        self.individual = configs.individual
        assert self.height % self.patch_size == 0, "Height must be divisible by patch size"
        assert self.width % self.patch_size == 0, "Width must be divisible by patch size"

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.individual)

        self.encoder = MSPSTTEncoder(
                    [
                        MSPSTTEncoderLayer(
                            VariableAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                in_chans=configs.enc_in,
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                learned_pe=False,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                pos_drop=configs.dropout,
                            ) if configs.individual else None,
                            MultiScalePeriodicAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                FourierLayer(
                                    seq_len=configs.seq_len,
                                    top_k=configs.top_k,
                                    d_model=configs.d_model,
                                    in_chans=configs.enc_in,
                                    img_size=(configs.height//self.patch_size, configs.width//self.patch_size),
                                    fuse_drop=configs.dropout,
                                    position_wise=configs.position_wise,
                                    individual=configs.individual,
                                ),
                                seq_len=configs.seq_len,
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                learned_pe=False,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                pos_drop=configs.dropout,
                                position_wise=False,
                                individual=False,
                            ),
                            WindowAttentionLayer(
                                MyWindowAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    window_size=window_size,
                                    qkv_bias=True,
                                    attn_drop=configs.dropout,
                                    proj_drop=configs.dropout,
                                    qk_norm=False,
                                    output_attention=configs.output_attention,
                                ),
                                input_resolution=(configs.height // self.patch_size, configs.width // self.patch_size),
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else shift_size,
                                always_partition=False,
                                dynamic_mask=False,
                            ),
                            d_model=configs.d_model,
                            d_ff=configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                            pre_norm=configs.pre_norm,
                            is_parallel=configs.is_parallel,
                        )
                        for i in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )

        # self.patch_recovery = PatchRecovery(self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.individual)
        self.patch_recovery = PatchInflated(self.patch_size, configs.enc_in, configs.d_model, stride=2, padding=1, output_padding=1, individual=configs.individual)

    def forward_core(self, x_enc):
        # x_enc [B, T, H, W, C]

        # patch embedding
        x_embed = self.patch_embed(x_enc) # -> [B, T, C, H, W, D] C=1 if individual=True else enc_in
        
        # encoding
        enc_out, _ = self.encoder(x_embed) # -> [B, T, C, H, W, D]

        # decoding
        dec_out = self.patch_recovery(enc_out) # -> [B, T, H, W, C]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        if self.seq_len >= self.pred_len:
            return self.forward_core(x_enc)[:,:self.pred_len], None
        else:
            d = self.pred_len // self.seq_len 
            r = self.pred_len % self.seq_len
            ys = []
            x = x_enc
            for i in range(d):
                x = self.forward_core(x)
                ys.append(x)
            if r > 0:
                x = self.forward_core(x)
                ys.append(x[:,:r])
            y = torch.cat(ys, dim=1)
            return y, None