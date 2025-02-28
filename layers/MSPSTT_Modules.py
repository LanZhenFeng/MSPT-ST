from typing import List, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil, log
from einops import rearrange
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.layers import Mlp, LayerNorm2d, use_fused_attn, to_2tuple

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

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # 计算词向量之后，两两分组，每组对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # end 一般指句子的最大长度
    t = torch.arange(seq_len, device=freqs.device)  # type: ignore
    # 计算 t和 freqs 的外积

    # seq_len * (dim // 2)
    freqs = torch.outer(t, freqs).float()  # type: ignore

    # torch.polar 
    # seq_len * (dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
    
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_len = xq.shape[1]
    xk_len = xk.shape[1]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis[:xq_len], xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis[:xk_len], xk_)
    # freqs_cis [1, seq_len, dim // 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FullAttention(nn.Module):
    r"""
    
    The implementation of full attention originally from Attention is All You Need.

    Args:
        d_model (int): The dimension of queries, keys and values.
        n_heads (int): The number of heads in multi-head attention.
        attn_drop (float): The dropout rate of attention weights.
        is_causal (bool): Whether the attention is causal or not.
        output_attention (bool): Whether to output the attention weights.

    """
    def __init__(
            self,
            d_model: int = 512,
            n_heads: int = 8,
            attn_drop: float = 0.,
            is_causal: bool = False,
            output_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, 'dim should be divisible by num_heads'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_drop = nn.Dropout(attn_drop)
        self.is_causal = is_causal
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
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
            attn_weight += attn_bias
            attn_weight = attn_weight.softmax(dim=-1)
            attn_weight = self.attn_drop(attn_weight)
            if self.output_attention:
                return attn_weight @ values, attn_weight
            else:
                return attn_weight @ values, None


class Gate(nn.Module):
    r"""

    The implementation of the gate mechanism in MSPSTT to select experts that are responsible for different periodic scales.

    Args:
        seq_len (int): The length of the sequence.
        top_k (int): The number of experts to be selected.
        d_model (int): The dimension of the input tensor.
        img_size (Tuple[int, int]): The size of the input image.
        use_conv (bool): Whether to use convolutional layers in the gate.
        use_linear (bool): Whether to use linear layers in the gate.
        use_pool (bool): Whether to use pooling layers in the gate.
        dropout (float): The dropout rate of the convolutional layers and linear layers.

    """

    def __init__(
            self,
            seq_len: int,
            top_k: int,
            height: int,
            width: int,
            d_model: int,
            use_conv: bool = False,
            use_linear: bool = False,
            dropout: float = 0.,
    ):
        super(Gate, self).__init__()
        assert sum([use_conv, use_linear]) <= 1, 'Choose at most one of the options: use_conv, use_linear'
        self.seq_len = seq_len
        self.top_k = top_k
        self.height = height
        self.width = width
        self.num_freqs = seq_len // 2
        
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)

        # convolutional layers
        if use_conv and not use_linear:
            self.conv = nn.Conv2d(in_channels=d_model, out_channels=d_model * 4, kernel_size=(height, width)) # [B, T, H, W, D] -> [B, T, 1, 1, D*4]
            self.conv_drop = nn.Dropout(dropout)

        if use_linear and not use_conv:
            self.linear = nn.Linear(height * width * d_model, d_model * 4)
            self.linear_drop = nn.Dropout(dropout)

        if not use_conv and not use_linear:
            self.pool = nn.AdaptiveAvgPool2d((1, 1)) # [B, T, H, W, D] -> [B, T, 1, 1, D]

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

        # Normal distribution
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def get_segment_sizes(self, seq_len):
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
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
        if hasattr(self, 'conv'):
            x = self.conv(x)
            x = self.conv_drop(x)
            x = x.flatten(1)
        elif hasattr(self, 'linear'):
            x = x.flatten(1)
            x = self.linear(x)
            x = self.linear_drop(x)
        else:
            x = self.pool(x).flatten(1)
        x = rearrange(x, '(b t) d -> b t d', t=T)

        # fft
        xf = torch.fft.rfft(x, dim=-2, norm='ortho')[:, 1:]  # [B, T//2, D]
        amp = torch.abs(xf).mean(dim=-1)  # [B, T//2]
        clean_logits = amp @ self.w_gate # [B, Ps]

        if training:
            raw_noise_stddev = amp @ self.w_noise # [B, Ps]
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits  # [B,  Ps]
        else:
            logits = clean_logits  # [B, Ps]

        top_logits, top_indices = torch.topk(logits, self.top_k+1, dim=-1)  # [B, top_k], [B, top_k]
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(logits, requires_grad=True)  # [B, Ps]
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        if self.top_k < len(self.segment_sizes) and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class PeriodicAttentionLayer(nn.Module):
    def __init__(
            self, 
            attention: nn.Module = FullAttention,
            segment_size: int = 2,
            d_model: int = 512,
            d_scale: int = 2,
            n_heads: int = 8,
            qkv_bias: bool = False, 
            qk_norm: bool = False, 
            proj_bias: bool = True, 
            proj_drop: float = 0.,
            is_rotary: bool = False,
    ):
        super(PeriodicAttentionLayer, self).__init__()
        assert d_model*d_scale % n_heads == 0, 'd_model*d_scale should be divisible by n_heads'
        self.segment_size = segment_size
        self.inner_attention = attention
        self.n_heads = n_heads
        self.in_dim = d_model * segment_size
        self.hid_dim = d_model * d_scale # d_scale = average size of the segments
        self.out_dim = d_model * segment_size
        self.head_dim = self.hid_dim // n_heads
        self.q_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.k_proj= nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.in_dim, self.hid_dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.o_proj = nn.Linear(self.hid_dim, self.out_dim, bias=proj_bias)
        self.o_proj_drop = nn.Dropout(proj_drop)

        self.is_rotary = is_rotary # rotary embedding

    def transform_input(self, x):
        T = x.shape[1]
        x = rearrange(x, 'b t h w d -> (b h w) t d')
        padding_len = self.segment_size - T % self.segment_size if T % self.segment_size != 0 else 0
        # x = F.pad(x, (0, 0, 0, padding_len), mode='replicate')
        x = F.pad(x, (0, 0, 0, padding_len), mode='constant', value=0)
        x = x.unfold(1, self.segment_size, self.segment_size)
        x = rearrange(x, 'b n d p -> b n (p d)')
        return x

    def forward(self, queries, keys, values, freqs_cis=None, attn_mask=None):
        # x [B, T, H, W, D]
        if queries.shape[0] == 0:
            return queries

        B, T, H, W, D = queries.shape
        # rearrange and segment
        queries = self.transform_input(queries)
        keys = self.transform_input(keys)
        values = self.transform_input(values)

        # qkv projection
        queries, keys, values = self.q_proj(queries), self.k_proj(keys), self.v_proj(values)
        # rotary embedding
        if self.is_rotary:
            assert freqs_cis is not None, 'freqs_cis should not be None for rotary embedding'
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis)
        queries, keys, values = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (queries, keys, values))
        queries, keys = self.q_norm(queries), self.k_norm(keys)

        # attention
        x, attn = self.inner_attention(queries, keys, values, attn_mask)

        # projection and reshape
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.o_proj(x)
        x = self.o_proj_drop(x)
        x = rearrange(x, 'b n (p d) -> b (n p) d', p=self.segment_size)[:,:T]
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x


class MultiScalePeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, top_k, img_size, d_model, n_heads, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., use_conv=False, use_linear=False, is_rotary=False, dropout=0.):
        super(MultiScalePeriodicAttentionLayer, self).__init__()
        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        self.gate_layer = Gate(seq_len=seq_len, top_k=top_k, height=img_size[0], width=img_size[1], d_model=d_model, use_conv=use_conv, use_linear=use_linear, dropout=dropout)
        self.segment_sizes = self.gate_layer.segment_sizes

        d_scale = np.mean(self.segment_sizes).astype(int)
        self.attention_layers = nn.ModuleList()
        for segment_size in self.segment_sizes:
            self.attention_layers.append(
                PeriodicAttentionLayer(
                    attention,
                    segment_size,
                    d_model,
                    d_scale,
                    n_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop,
                    is_rotary=is_rotary,
                )
            )
    
    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, freqs_cis=None, attn_mask=None, loss_coef=1e-2):
        # queries, keys, values [B, T, H, W, D]
        B, T, H, W, D = x.shape

        # gating
        gates, load = self.gate_layer(x, training=self.training) # [B, Ps]

        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef

        # dispatch
        xs = dispatch(x, gates) # Ps * [B, T, H, W, D]

        # multi-branch attention
        xs = [attention_layer(x_item, x_item, x_item, freqs_cis, attn_mask=attn_mask) for x_item, attention_layer in zip(xs, self.attention_layers)]

        # combine
        x = combine(xs, gates)

        return x, balance_loss


class WindowAttentionLayer(nn.Module):
    r"""Window Attention Layer of Swin Transformer. """
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
        B, T, H, W, D = x.shape

        # reshape
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

        # reshape back
        x = rearrange(x, '(b t) h w d -> b t h w d', b=B, t=T)

        return x


class TemporalAttentionLayer(nn.Module):
    def __init__(
            self,
            attention: nn.Module = FullAttention,
            d_model: int = 512,
            n_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True, 
            proj_drop: float = 0.,
            is_rotary: bool = False,
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj= nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.o_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.o_proj_drop = nn.Dropout(proj_drop)

        self.is_rotary = is_rotary # rotary embedding

    def forward(self, queries, keys, values, freqs_cis=None, attn_mask=None):
        # x [B, T, H, W, D]
        B, T, H, W, D = queries.shape

        # reshape
        queries, keys, values = map(lambda x: rearrange(x, 'b t h w d -> (b h w) t d'), (queries, keys, values))

        # qkv projection
        queries, keys, values = self.q_proj(queries), self.k_proj(keys), self.v_proj(values)
        # rotary embedding
        if self.is_rotary:
            assert freqs_cis is not None, 'freqs_cis should not be None for rotary embedding'
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis)
        # reshape
        queries, keys, values = map(lambda x: rearrange(x, 'b t (n d) -> b n t d', n=self.n_heads), (queries, keys, values))
        # qurery and key normalization
        queries, keys = self.q_norm(queries), self.k_norm(keys)

        # attention
        x, attn = self.inner_attention(queries, keys, values, attn_mask)

        # reshape and projection
        x = rearrange(x, 'b n t d -> b t (n d)')
        x = self.o_proj(x)
        x = self.o_proj_drop(x)
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x