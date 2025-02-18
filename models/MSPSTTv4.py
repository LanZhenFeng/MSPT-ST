import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil, log
from einops import rearrange
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.layers import Mlp, LayerNorm2d, use_fused_attn, to_2tuple
from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

def dispatch(inp, gates):
    # sort experts
    _, index_sorted_experts = torch.nonzero(gates).sort(0)
    # get according batch index for each expert
    _batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
    _part_sizes = (gates > 0).sum(0).tolist()
    # check if _part_sizes is all zeros
    if sum(_part_sizes) == 0:
        print(f"All zero parts, gates: {gates}")
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
            attn_weight += attn_bias
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

        # weight initialization
        if learned_pe:
            nn.init.trunc_normal_(self.pos_embed, std=.02)

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
        # get the number of convolutional layers
        self.num_conv_layers, self.hw = self.get_num_conv_layers()
        # convolutional layers
        self.Convs = nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.Convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2), 
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(inplace=True),
                    # LayerNorm2d(d_model),
                    # nn.SiLU(inplace=True)
                    # nn.GELU()
                )
            )

        self.in_chans = in_chans if individual else 1
        input_size = self.in_chans * d_model if position_wise else self.in_chans * d_model * self.hw
        self.fuse_proj = nn.Linear(input_size, d_model)
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

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        from torch.distributions.normal import Normal
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
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
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape
        assert C == self.in_chans, 'input channels should be equal to in_chans'

        # embed & fuse
        if self.position_wise:
            x = rearrange(x, 'b t c h w d -> (b h w) t (c d)')
        else:
            x = rearrange(x, 'b t c h w d -> (b t c) d h w')
            for conv in self.Convs:
                x = conv(x)
            x = rearrange(x, '(b t c) d h w -> b t (c h w d)', c=C, t=T)
        x = self.fuse_proj(x)
        x = self.fuse_drop(x)

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

        top_weights, top_indices = torch.topk(weights, self.top_k+1, dim=-1)  # [B, top_k], [B, top_k]
        top_k_weights = top_weights[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = F.softmax(top_k_weights, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(weights, requires_grad=True)  # [B, Ps]
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        noisy_gating = True

        if noisy_gating and self.top_k < len(self.segment_sizes) and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_weights)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class PeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, segment_size, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0., position_wise=False, individual=False):
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
        self.qkv = nn.Linear(self.in_dim, self.hid_dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(self.hid_dim, self.out_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.position_wise = position_wise
        self.individual = individual

        self.learned_pe = learned_pe
        num_segments = seq_len // segment_size if seq_len % segment_size == 0 else seq_len // segment_size + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_segments, self.in_dim)) if learned_pe else PositionalEmbedding(self.in_dim)
        self.pos_drop = nn.Dropout(pos_drop)

        # weight initialization
        if learned_pe:
            nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D] or [BHW, T, C, D]
        if x.shape[0] == 0:
            return x, None

        if self.position_wise:
            B, T, C, D = x.shape
            x = rearrange(x, 'bhw t c d -> (bhw c) t d')
        else:
            B, T, C, H, W, D = x.shape
            x = rearrange(x, 'b t c h w d -> (b c h w) t d')

        # segment
        padding_len = self.segment_size - T % self.segment_size if T % self.segment_size != 0 else 0
        x = F.pad(x, (0, 0, 0, padding_len), mode='replicate')
        x = x.unfold(1, self.segment_size, self.segment_size)
        x = rearrange(x, 'b n d p -> b n (p d)')

        # positional embedding
        if self.learned_pe:
            x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x + self.pos_embed(x))

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
            x = rearrange(x, '(bhw c) t d -> bhw t c d', c=C)
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

    def forward(self, x, attn_mask=None, tau=None, delta=None, loss_coef=1e-2):
        # x [B, T, C, H, W, D]
        B, T, C, H, W, D = x.shape

        # gating
        gates, load = self.gate_layer(x, training=self.training) # [B, Ps]

        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef

        # re-arrange
        if self.position_wise:
            x = rearrange(x, 'b t c h w d -> (b h w) t c d')

        # dispatch
        xs = dispatch(x, gates) # Ps * [B, T, C, H, W, D] or Ps * [BHW, T, C, D]
        
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

        # re-arrange
        if self.position_wise:
            x = rearrange(x, '(b h w) t c d -> b t c h w d', h=H, w=W)

        return x, attns, balance_loss

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
        assert isinstance(attention_s, nn.ModuleList), 'attention_s should be a nn.ModuleList'
        attention_s_depth = len(attention_s)
        activation = nn.ReLU if activation == "relu" else nn.GELU
        self.mlp_v = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout) if attention_v is not None else None
        self.mlp_t = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.mlp_s = nn.ModuleList([Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout) for _ in range(attention_s_depth)])
        self.norm1 = nn.LayerNorm(d_model) # for variable attention
        self.norm2 = nn.LayerNorm(d_model) # for variable mlp
        self.norm3 = nn.LayerNorm(d_model) # for temporal attention
        self.norm4 = nn.LayerNorm(d_model) # for temporal mlp
        self.norm5 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(attention_s_depth)]) # for spatial attention
        self.norm6 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(attention_s_depth)]) # for spatial mlp
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.is_parallel = is_parallel

        self.fusion_st = Mlp(in_features=d_model*2, hidden_features=d_model*4, out_features=d_model, act_layer=activation, drop=dropout)
        self.norm7 = nn.LayerNorm(d_model*2) if pre_norm else nn.LayerNorm(d_model) # for fusion

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

            # variable mlp
            res = x
            if self.pre_norm:
                x = self.norm2(x)
            x = self.mlp_v(x)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm2(x)
        else:
            attn_v = None

        if self.is_parallel:
            ## parallel spatial attention and temporal attention
            # temporal attention
            res_t = x
            if self.pre_norm:
                x_t = self.norm3(x)
            else:
                x_t = x
            x_t, attn_t, balance_loss = self.attention_t(x_t, attn_mask=attn_mask, tau=tau, delta=delta)
            x_t = res_t + self.dropout(x_t)
            if not self.pre_norm:
                x_t = self.norm3(x_t)

            # temporal mlp
            res_t = x_t
            if self.pre_norm:
                x_t = self.norm4(x_t)
            x_t = self.mlp_t(x_t)
            x_t = res_t + self.dropout(x_t)
            if not self.pre_norm:
                x_t = self.norm4(x_t)

            # spatial attention
            x_s = x
            for attention_s, mlp_s, norm_attn, norm_mlp in zip(self.attention_s, self.mlp_s, self.norm5, self.norm6):
                res_s = x_s
                if self.pre_norm:
                    x_s = norm_attn(x_s)
                x_s, attn_s = attention_s(x_s)
                x_s = res_s + self.dropout(x_s)
                if not self.pre_norm:
                    x_s = norm_attn(x_s)

                # spatial mlp
                res_s = x_s
                if self.pre_norm:
                    x_s = norm_mlp(x_s)
                x_s = mlp_s(x_s)
                x_s = res_s + self.dropout(x_s)
                if not self.pre_norm:
                    x_s = norm_mlp(x_s)

            # combine
            x = torch.cat([x_t, x_s], dim=-1)
            # x = self.fusion_st(x)
            if self.pre_norm:
                x = self.norm7(x)
            x = self.fusion_st(x)
            x = x + self.dropout(x)
            if not self.pre_norm:
                x = self.norm7(x)

        else:
            ## serial spatial attention and temporal attention
            res = x
            # temporal attention
            if self.pre_norm:
                x = self.norm3(x)
            x, attn_t, balance_loss = self.attention_t(x, attn_mask=attn_mask, tau=tau, delta=delta)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm3(x)

            # temporal mlp
            res = x
            if self.pre_norm:
                x = self.norm4(x)
            x = self.mlp_t(x)
            x = res + self.dropout(x)
            if not self.pre_norm:
                x = self.norm4(x)

            # spatial attention
            for attention_s, mlp_s, norm_attn, norm_mlp in zip(self.attention_s, self.mlp_s, self.norm5, self.norm6):
                res = x
                if self.pre_norm:
                    x = norm_attn(x)
                x, attn_s = attention_s(x)
                x = res + self.dropout(x)
                if not self.pre_norm:
                    x = norm_attn(x)

                # spatial mlp
                res = x
                if self.pre_norm:
                    x = norm_mlp(x)
                x = mlp_s(x)
                x = res + self.dropout(x)
                if not self.pre_norm:
                    x = norm_mlp(x)

        return x, (attn_v, attn_t, attn_s), balance_loss
    

class MSPSTTEncoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None):
        super(MSPSTTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, C, H, W, D]
        attns = []
        balance_loss = 0.

        for encoder_layer in self.encoder_layers:
            x, attn, aux_loss = encoder_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            balance_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, balance_loss


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, individual=False):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.individual = individual
        self.in_chans = 1 if individual else in_chans
        self.proj = nn.Conv2d(self.in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, H, W, C] -> [B, T, C', H', W', D]
        B, T, H, W, C = x.shape
        x = rearrange(x, 'b t h w (c d) -> (b t c) d h w', d=self.in_chans)
        x = self.proj(x)
        x = rearrange(x, '(b t c) d h w -> b t c h w d', b=B, t=T)
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
    def __init__(self, configs, depths=[2, 2, 2, 2], window_size=4): 
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

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.individual)
        
        self.temporal_embedding = TemporalEmbedding(d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq) if configs.embed != 'timeF' else TimeFeatureEmbedding(d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq)
        self.temporal_embedding_dropout = nn.Dropout(configs.dropout)

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
                            nn.ModuleList(
                                [
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
                                        input_resolution=(configs.height//self.patch_size, configs.width//self.patch_size),
                                        window_size=window_size,
                                        shift_size=0 if (j % 2 == 0) else window_size//2,
                                        always_partition=False,
                                        dynamic_mask=False,
                                    ) for j in range(depths[i])
                                ]
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

        self.patch_recovery = PatchRecovery(self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.individual)
        # self.patch_recovery = PatchInflated(self.patch_size, configs.enc_in, configs.d_model, stride=2, padding=1, output_padding=1, individual=configs.individual)

    def forward_core(self, x_enc, x_mark_enc):
        # x_enc [B, T, H, W, C]

        # patch embedding
        x_enc = self.patch_embed(x_enc) # -> [B, T, C, H, W, D] C=1 if individual=True else enc_in

        # temporal embedding
        x_enc = self.temporal_embedding_dropout(x_enc + self.temporal_embedding(x_mark_enc).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2))
        
        # encoding
        enc_out, _, balance_loss = self.encoder(x_enc) # -> [B, T, C, H, W, D]

        # patch recovery
        dec_out = self.patch_recovery(enc_out) # -> [B, T, H, W, C]

        return dec_out, balance_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        if self.seq_len >= self.pred_len:
            dec_out, balance_loss = self.forward_core(x_enc, x_mark_enc)
            return dec_out[:,:self.pred_len], balance_loss
        else:
            d = self.pred_len // self.seq_len 
            r = self.pred_len % self.seq_len
            x_mark = torch.cat([x_mark_enc, x_mark_dec], dim=1)
            ys = []
            x = x_enc
            for i in range(d):
                time_mark = x_mark[:,i*self.seq_len:(i+1)*self.seq_len]
                x = self.forward_core(x, time_mark)
                ys.append(x)
            if r > 0:
                time_mark = x_mark[:,d*self.seq_len:]
                x = self.forward_core(x, x_mark_enc)
                ys.append(x[:,:r])
            y = torch.cat(ys, dim=1)
            return y, None