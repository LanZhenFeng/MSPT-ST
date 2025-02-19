from typing import Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil, log
from einops import rearrange
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.models.vision_transformer import Attention as VariableAttention
from timm.layers import Mlp, LayerNorm2d, use_fused_attn, to_2tuple
from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

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


def combine(expert_out, gates, multiply_by_gates=True):
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
        dims_stitched = stitched.dim()
        dims_nonzero_gates = _nonzero_gates.dim()
        for i in range(dims_stitched - dims_nonzero_gates):
            _nonzero_gates = _nonzero_gates.unsqueeze(-1)
        stitched = stitched * _nonzero_gates
    zeros = torch.zeros(gates.size(0), *expert_out[-1].shape[1:], requires_grad=True, device=stitched.device)
    # combine samples that have been processed by the same k experts
    combined = zeros.index_add(0, _batch_index, stitched.float())
    # add eps to all zero values in order to avoid nans when going back to log space
    combined[combined == 0] = np.finfo(float).eps
    # back to log space
    return combined.log()


class FullAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 attn_drop: float = 0.,
                 is_causal: bool = False,
                 output_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        self.output_attention = output_attention
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.fused_attn = use_fused_attn()

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, H, L, E = queries.shape
        _, H, S, D = values.shape

        if self.fused_attn and not self.output_attention:
            return F.scaled_dot_product_attention(queries, keys, values, attn_mask,dropout_p=self.attn_drop.p if self.training else 0., is_causal=self.is_causal), None
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
            attn_weight += attn_bias
            attn_weight = attn_weight.softmax(dim=-1)
            attn_weight = self.attn_drop(attn_weight)
            if self.output_attention:
                return attn_weight @ values, attn_weight
            else:
                return attn_weight @ values, None


class Gate(nn.Module):
    def __init__(self,
                 seq_len: int,
                 top_k: int,
                 d_model: int,
                 img_size: int,
                 fuse_drop: float = 0.):
        super(Gate, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_freqs = seq_len // 2
        self.height, self.width = img_size
        
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)
        # get the number of convolutional layers
        num_conv_layers, num_squeeze_tokens = self.get_num_conv_layers()
        # convolutional layers
        self.Convs = nn.ModuleList()
        out_dim = d_model
        for i in range(num_conv_layers):
            in_dim = d_model * 2 ** i
            out_dim = d_model * 2 ** (i + 1)
            self.Convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2), 
                    LayerNorm2d(out_dim),
                    # nn.ReLU(inplace=True)
                    nn.GELU()
                )
            )

        self.fuse_proj = nn.Linear(out_dim * num_squeeze_tokens, out_dim)
        self.fuse_drop = nn.Dropout(fuse_drop)

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def get_num_conv_layers(self):
        ks = 2
        num_layers = 0
        height, width = self.height, self.width
        while height % ks == 0 and width % ks == 0:
            num_layers += 1
            height //= ks
            width //= ks
        return num_layers, height*width

    def get_segment_sizes(self, seq_len):
        # get the period list, first element is inf
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
        # segment_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        # print(f"Segment sizes: {segment_sizes}")
        return segment_sizes

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        from torch.distributions.normal import Normal
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x, training, noise_epsilon=1e-2):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape

        # embed & fuse
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        for conv in self.Convs:
            x = conv(x)
        x = rearrange(x, '(b t) d h w -> b t (h w d)', t=T)
        x = self.fuse_proj(x)
        x = self.fuse_drop(x)

        # fft
        xf = torch.fft.rfft(x, dim=-2, norm='ortho')[:, 1:]  # [B, T//2, D]
        amp = torch.abs(xf).mean(dim=-1)  # [B, T//2]
        clean_logits = amp @ self.w_gate # [B, Ps]

        if training:
            raw_noise_stddev = amp @ self.w_noise # [B, Ps]
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            logits = noisy_logits  # [B,  Ps]
        else:
            logits = clean_logits  # [B, Ps]

        top_logits, top_indices = torch.topk(logits, self.top_k+1, dim=-1)  # [B, top_k], [B, top_k]
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(logits, requires_grad=True)  # [B, Ps]
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        noisy_gating = True

        if noisy_gating and self.top_k < len(self.segment_sizes) and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class Gate2(nn.Module):
    def __init__(self,
                 seq_len: int,
                 top_k: int,
                 d_model: int,
                 img_size: int,
                 fuse_drop: float = 0.):
        super(Gate2, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_freqs = seq_len // 2
        self.height, self.width = img_size
        
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)
        # get the number of convolutional layers
        num_conv_layers, num_squeeze_tokens = self.get_num_conv_layers()
        # convolutional layers
        self.Convs = nn.ModuleList()
        out_dim = d_model
        for i in range(num_conv_layers):
            in_dim = d_model * 2 ** i
            out_dim = d_model * 2 ** (i + 1)
            self.Convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2), 
                    LayerNorm2d(out_dim),
                    # nn.ReLU(inplace=True)
                    nn.GELU()
                )
            )

        self.fuse_proj = nn.Linear(out_dim * num_squeeze_tokens, out_dim)
        self.fuse_drop = nn.Dropout(fuse_drop)

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def get_num_conv_layers(self):
        ks = 2
        num_layers = 0
        height, width = self.height, self.width
        while height % ks == 0 and width % ks == 0:
            num_layers += 1
            height //= ks
            width //= ks
        return num_layers, height*width

    def get_segment_sizes(self, seq_len):
        # get the period list, first element is inf
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
        # segment_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        # print(f"Segment sizes: {segment_sizes}")
        return segment_sizes

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        from torch.distributions.normal import Normal
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x, training, noise_epsilon=1e-2):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape

        # embed & fuse
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        for conv in self.Convs:
            x = conv(x)
        x = rearrange(x, '(b t) d h w -> b t (h w d)', t=T)
        x = self.fuse_proj(x)
        x = self.fuse_drop(x)

        # fft
        xf = torch.fft.rfft(x, dim=-2, norm='ortho')[:, 1:]  # [B, T//2, D]
        amp = torch.abs(xf).mean(dim=-1)  # [B, T//2]
        clean_logits = amp @ self.w_gate # [B, Ps]

        if training:
            raw_noise_stddev = amp @ self.w_noise # [B, Ps]
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            logits = noisy_logits  # [B,  Ps]
        else:
            logits = clean_logits  # [B, Ps]

        top_logits, top_indices = torch.topk(logits, self.top_k+1, dim=-1)  # [B, top_k], [B, top_k]
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(logits, requires_grad=True)  # [B, Ps]
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        noisy_gating = True

        if noisy_gating and self.top_k < len(self.segment_sizes) and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class AttentionLayer(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 d_model: int,
                 dim_scale: int = 1,
                 n_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 proj_bias: bool = True,
                 proj_drop: float = 0.):
        super(AttentionLayer, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.inner_attention = attention
        self.n_heads = n_heads
        self.in_dim = d_model
        self.hid_dim = d_model // dim_scale
        self.out_dim = d_model
        self.head_dim = self.hid_dim // n_heads
        self.q_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.k_proj= nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.o_proj = nn.Linear(self.hid_dim, self.out_dim, bias=proj_bias)
        self.o_proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # x [BHW, T, D]
        # if queries.shape[0] == 0:
        #     return queries, None

        # qkv projection
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.n_heads)
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.n_heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.n_heads)
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
        x = self.o_proj(x)
        x = self.o_proj_drop(x)

        return x, attn


class PeriodicAttentionLayer(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 seq_len: int,
                 segment_size: int,
                 d_model: int,
                 n_heads: int = 8,
                 learned_pe: bool = False,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 proj_bias: bool = True,
                 proj_drop: float = 0.,
                 pos_drop: float = 0.,):
        super(PeriodicAttentionLayer, self).__init__()
        assert segment_size*d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.seq_len = seq_len
        self.segment_size = segment_size
        self.inner_attention = attention
        self.n_heads = n_heads
        self.in_dim = d_model * segment_size
        self.hid_dim = d_model * (segment_size//2)
        self.out_dim = d_model * segment_size
        self.head_dim = self.hid_dim // n_heads
        self.q_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.k_proj= nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.o_proj = nn.Linear(self.hid_dim, self.out_dim, bias=proj_bias)
        self.o_proj_drop = nn.Dropout(proj_drop)

        self.learned_pe = learned_pe
        num_segments = seq_len // segment_size if seq_len % segment_size == 0 else seq_len // segment_size + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_segments, self.in_dim)) if learned_pe else PositionalEmbedding(self.in_dim)
        self.pos_drop = nn.Dropout(pos_drop)

        # weight initialization
        if learned_pe:
            nn.init.trunc_normal_(self.pos_embed, std=.02)

    def transform_input(self, x):
        T = x.shape[1]
        x = rearrange(x, 'b t h w d -> (b h w) t d')
        padding_len = self.segment_size - T % self.segment_size if T % self.segment_size != 0 else 0
        x = F.pad(x, (0, 0, 0, padding_len), mode='replicate')
        x = x.unfold(1, self.segment_size, self.segment_size)
        x = rearrange(x, 'b n d p -> b n (p d)')
        # add position embedding
        if self.learned_pe:
            x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x + self.pos_embed(x))
        return x

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        if queries.shape[0] == 0:
            return queries, None

        B, T, H, W, D = queries.shape
        # rearrange and segment
        queries = self.transform_input(queries)
        keys = self.transform_input(keys)
        values = self.transform_input(values)

        # qkv projection
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.n_heads)
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.n_heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.n_heads)
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
        x = self.o_proj(x)
        x = self.o_proj_drop(x)
        x = rearrange(x, 'b n (p d) -> b (n p) d', p=self.segment_size)[:,:T]
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x, attn


class MultiScalePeriodicAttentionLayer(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 seq_len: int,
                 top_k: int,
                 d_model: int,
                 n_heads: int = 8,
                 img_size: Tuple[int, int] = (32, 32),
                 learned_pe: bool = False,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 proj_bias: bool = True,
                 proj_drop: float = 0.,
                 pos_drop: float = 0.,
                 gate_drop: float = 0.,):
        super(MultiScalePeriodicAttentionLayer, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.gate_layer = Gate(seq_len, top_k, d_model, img_size, gate_drop)
        self.segment_sizes = self.gate_layer.segment_sizes

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
                )
            )
    
    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, attn_mask=None, tau=None, delta=None, loss_coef=1e-2):
        # x [B, T, C, H, W, D]
        B, T, H, W, D = x.shape

        # gating
        gates, load = self.gate_layer(x, training=self.training) # [B, Ps]

        # dispatch
        xs = dispatch(x, gates) # Ps * [B, T, H, W, D]

        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        
        # multi-branch attention
        # attns = []
        # for i, (x_item, attention_layer) in enumerate(zip(xs, self.attention_layers)):
        #     x_item, attn = attention_layer(x_item, attn_mask=attn_mask, tau=tau, delta=delta)
        #     xs[i] = x_item
        #     attns.append(attn)
        xs = [attention_layer(x_item, attn_mask=attn_mask, tau=tau, delta=delta)[0] for x_item, attention_layer in zip(xs, self.attention_layers)]
        attns = None

        # combine
        x = combine(xs, gates)

        return x, attns, balance_loss


class WindowAttentionLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            img_size: Tuple[int, int] = (32, 32),
            window_size: int = 7,
            shift_size: int = 0,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            always_partition: bool = False, 
            dynamic_mask: bool = False
    ):
        super(WindowAttentionLayer, self).__init__()
        self.attention = WindowAttention(d_model, n_heads, window_size=window_size, attn_drop=attn_drop, proj_drop=proj_drop)
        self.input_resolution = img_size
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
        self.attention.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        
        # re-arrange
        x = rearrange(x, 'b t h w d -> (b t) h w d')

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
        attn_windows = self.attention(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, D

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
        x = rearrange(x, '(b t) h w d -> b t h w d', b=B, t=T)

        return x, None


class MultiScalePeriodicSpatialTemporalBlock(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 seq_len: int,
                 top_k: int,
                 d_model: int,
                 n_heads: int = 8,
                 img_size: Tuple[int, int] = (32, 32),
                 window_size: int = 7,
                 shift_size: int = 0,
                 learned_pe: bool = False,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_bias: bool = True,
                 proj_drop: float = 0.,
                 mlp_drop: float = 0.,
                 pos_drop: float = 0.,
                 gate_drop: float = 0.,
                 dropout: float = 0.,
                 activation: str = "relu",
                 always_partition: bool = False,
                 dynamic_mask: bool = False,
                 pre_norm: bool = False):
        super(MultiScalePeriodicSpatialTemporalBlock, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        activation = nn.ReLU if activation == "relu" else nn.GELU
        self.gate_layer = Gate(seq_len, top_k, d_model, img_size, gate_drop)
        self.segment_sizes = self.gate_layer.segment_sizes

        self.attention_t_layers = nn.ModuleList()
        self.attention_s_layers = nn.ModuleList()
        self.mlp_t_layers = nn.ModuleList()
        self.mlp_s_layers = nn.ModuleList()
        self.norm_attn_t_layers = nn.ModuleList()
        self.norm_mlp_t_layers = nn.ModuleList()
        self.norm_attn_s_layers = nn.ModuleList()
        self.norm_mlp_s_layers = nn.ModuleList()
        for segment_size in self.segment_sizes:
            dim = d_model * segment_size
            self.attention_t_layers.append(
                AttentionLayer(
                    attention,
                    dim,
                    dim_scale=2,
                    n_heads=n_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop
                )
            )

            self.attention_s_layers.append(
                WindowAttentionLayer(
                    dim,
                    n_heads=n_heads,
                    img_size=img_size,
                    window_size=window_size,
                    shift_size=shift_size,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    always_partition=always_partition,
                    dynamic_mask=dynamic_mask
                )
            )
            self.mlp_t_layers.append(Mlp(in_features=dim, hidden_features=dim * 4, act_layer=activation, drop=mlp_drop))
            self.mlp_s_layers.append(Mlp(in_features=dim, hidden_features=dim * 4, act_layer=activation, drop=mlp_drop))
            self.norm_attn_t_layers.append(nn.LayerNorm(dim))
            self.norm_mlp_t_layers.append(nn.LayerNorm(dim))
            self.norm_attn_s_layers.append(nn.LayerNorm(dim))
            self.norm_mlp_s_layers.append(nn.LayerNorm(dim))
        
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

        self.learned_pe = learned_pe
        if learned_pe:
            self.pos_embeds = nn.ParameterList()
            for segment_size in self.segment_sizes:
                num_segments = seq_len // segment_size if seq_len % segment_size == 0 else seq_len // segment_size + 1
                self.pos_embeds.append(nn.Parameter(torch.zeros(1, num_segments, dim)))
                nn.init.trunc_normal_(self.pos_embeds[-1], std=.02)
        else:
            self.pos_embed = nn.ModuleList()
            for segment_size in self.segment_sizes:
                self.pos_embed.append(PositionalEmbedding(dim))
        
        self.pos_drop = nn.Dropout(pos_drop)

    def forward_one_branch(self, x, branch_idx, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        segment_size = self.segment_sizes[branch_idx]
        padding = segment_size - T % segment_size if T % segment_size != 0 else 0
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, padding), mode='replicate')
        x = x.unfold(1, segment_size, segment_size)
        x = rearrange(x, 'b f h w d p -> b f h w (d p)')

        # position embedding
        if self.learned_pe:
            x = self.pos_drop(x + self.pos_embeds[branch_idx])
        else:
            x = self.pos_drop(x + self.pos_embed[branch_idx](x))

        res = x
        if self.pre_norm:
            x = self.norm_attn_t_layers[branch_idx](x)
        x, attn_t = self.attention_t_layers[branch_idx](x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_attn_t_layers[branch_idx](x)
        
        res = x
        if self.pre_norm:
            x = self.norm_mlp_t_layers[branch_idx](x)
        x = self.mlp_t_layers[branch_idx](x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp_t_layers[branch_idx](x)

        res = x
        if self.pre_norm:
            x = self.norm_attn_s_layers[branch_idx](x)
        x, attn_s = self.attention_s_layers[branch_idx](x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_attn_s_layers[branch_idx](x)
        
        res = x
        if self.pre_norm:
            x = self.norm_mlp_s_layers[branch_idx](x)
        x = self.mlp_s_layers[branch_idx](x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp_s_layers[branch_idx](x)

        # reverse unfold
        x = rearrange(x, 'b f h w (d p) -> b (f p) h w d', p=segment_size)
        x = x[:, :T]

        return x
        

    def forward(self, x, attn_mask=None, tau=None, delta=None, loss_coef=1e-2):
        # x [B, T, C, H, W, D]
        B, T, H, W, D = x.shape

        # gating
        gates, load = self.gate_layer(x, training=self.training) # [B, Ps]

        # dispatch
        xs = dispatch(x, gates) # Ps * [B, T, H, W, D]

        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        
        # # multi-branch attention
        # attns = []
        # for i, (x_item, attention_layer) in enumerate(zip(xs, self.attention_layers)):
        #     x_item, attn = attention_layer(x_item, attn_mask=attn_mask, tau=tau, delta=delta)
        #     xs[i] = x_item
        #     attns.append(attn)
        xs = [self.forward_one_branch(x_item, i, attn_mask=attn_mask, tau=tau, delta=delta) for i, x_item in enumerate(xs)]
        attns = None

        # combine
        x = combine(xs, gates)

        return x, attns, balance_loss


class MSPSTTEncoderLayer(nn.Module):
    r"""

    Encoder layer for Multi-Scale Periodic Spatial-Temporal Transformer.
    Through swapping the order of the spatial attention (attention_s) and the temporal attention (attention_t), we can get two different versions of the encoder layer.

    """
    def __init__(
            self,
            attention_t: nn.Module,
            attention_s: nn.Module,
            d_model: int,
            d_ff: int = None,
            dropout: float = 0.,
            activation: str = "relu",
            pre_norm: bool = False
    ):
        super(MSPSTTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        activation = nn.ReLU if activation == "relu" else nn.GELU
        
        self.attention_t = attention_t
        self.attention_s = attention_s
        self.mlp_t = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.mlp_s = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.norm_attn_t = nn.LayerNorm(d_model)
        self.norm_mlp_t = nn.LayerNorm(d_model)
        self.norm_attn_s = nn.LayerNorm(d_model)
        self.norm_mlp_s = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.pre_norm = pre_norm

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
        res = x
        if self.pre_norm:
            x = self.norm_attn_s(x)
        x, attn_s = self.attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_attn_s(x)

        # res = x
        # if self.pre_norm:
        #     x = self.norm_mlp_s(x)
        # x = self.mlp_s(x)
        # x = res + self.dropout(x)
        # if not self.pre_norm:
        #     x = self.norm_mlp_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_attn_t(x)
        x, attn_t, balance_loss = self.attention_t(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_attn_t(x)

        res = x
        if self.pre_norm:
            x = self.norm_mlp_t(x)
        x = self.mlp_t(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp_t(x)

        return x, (attn_t, attn_s), balance_loss


class MSPSTTEncoder(nn.Module):
    def __init__(
            self, 
            encoder_layers: list,
            norm_layer: nn.Module = None
    ):
        super(MSPSTTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        aux_loss = 0.
        attns = []

        for encoder_layer in self.encoder_layers:
            x, attn, balance_loss = encoder_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            aux_loss += balance_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, aux_loss


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Tuple[int, int],
            patch_size: int,
            in_chans: int,
            embed_dim: int,
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, H, W, C] -> [B, T, H', W', D]
        B, T, H, W, C = x.shape
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) d h w -> b t h w d', t=T)
        return x


class PatchRecovery(nn.Module):
    r"""Patch Recovery Module"""
    def __init__(
            self, 
            img_size: Tuple[int, int],
            patch_size: int,
            in_chans: int,
            embed_dim: int,
            dropout: float = 0.
    ):
        super(PatchRecovery, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        x = self.proj(self.dropout(x))
        x = rearrange(x, '(b t) c h w -> b t h w c', t=T)
        return x


class DataEmbedding(nn.Module):
    def __init__(
            self,
            img_size: tuple,
            patch_size: int,
            in_chans: int = 3,
            d_model: int = 512,
            embed_type: str = 'fixed',
            freq: str = 'h',
            dropout: float = 0.1
    ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark).unsqueeze(-2).unsqueeze(-2)
        return self.dropout(x)


class Model(nn.Module):
    r'''
    
    Multi-Scale Periodic Spatio-Temporal Transformer ------ AR Architecture
    
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
        self.individual = configs.individual
        assert self.height % self.patch_size == 0, "Height must be divisible by patch size"
        assert self.width % self.patch_size == 0, "Width must be divisible by patch size"

        self.enc_embedding = DataEmbedding(img_size=self.img_size, patch_size=configs.patch_size, in_chans=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)

        self.encoder = MSPSTTEncoder(
                    [
                        MSPSTTEncoderLayer(
                            MultiScalePeriodicAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                seq_len=configs.seq_len,
                                top_k=configs.top_k,
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                img_size=(configs.height//self.patch_size, configs.width//self.patch_size),
                                learned_pe=False,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                pos_drop=configs.dropout,
                                gate_drop=configs.dropout,
                            ),
                            WindowAttentionLayer(
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                img_size=(configs.height//self.patch_size, configs.width//self.patch_size),
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size//2,
                                attn_drop=configs.dropout,
                                proj_drop=configs.dropout,
                                always_partition=False,
                                dynamic_mask=False,
                            ),
                            d_model=configs.d_model,
                            d_ff=configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                            pre_norm=configs.pre_norm,
                        )
                        for i in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )

        self.prediction_head = nn.Linear(self.seq_len*configs.d_model, configs.d_model)

        self.patch_recovery = PatchRecovery(self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.dropout)

        self.dropout = nn.Dropout(configs.dropout)

    def forward_core(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # encoding
        enc_out, _, aux_loss = self.encoder(enc_out)

        # prediction
        enc_out = rearrange(enc_out, 'b t h w d -> b h w (t d)')
        dec_out = self.prediction_head(self.dropout(enc_out)).unsqueeze(1) # -> [B, 1, H, W, C]
        # patch recovery
        dec_out = self.patch_recovery(dec_out) # -> [B, T, H, W, C]

        return dec_out, aux_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        # x_enc [batch, length, height, width, channel] -> [batch, length, channel, height, width]

        x_dec = x_dec[:, -self.pred_len:]
        x_mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:]], dim=1)

        predictions = []
        aux_loss = []

        prediction, balance_loss = self.forward_core(x_enc, x_mark[:, :self.seq_len])
        predictions.append(prediction)
        aux_loss.append(balance_loss)
        for t in range(self.pred_len - 1):
            add = mask_true[:, t] * x_dec[:, t] + (1 - mask_true[:, t]) * prediction.squeeze(1)
            x_enc = torch.cat([x_enc[:, 1:], add.unsqueeze(1)], dim=1)
            prediction, balance_loss = self.forward_core(x_enc, x_mark[:, t + 1:self.seq_len + t + 1])
            predictions.append(prediction)
            aux_loss.append(balance_loss)

        dec_out = torch.stack(predictions, dim=1)
        
        return dec_out, torch.stack(aux_loss).mean()