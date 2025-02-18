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
    # check if _part_sizes is all zeros
    # when all parts are zero, gates are zeros and nans
    # if sum(_part_sizes) == 0:
    #     print(f"All zero parts, gates: {gates}")
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
                 n_heads: int,
                 attn_drop: float = 0.,
                 is_causal: bool = False,
                 output_attention: bool = False):
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


class TemporalCrossAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, pred_len, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0.):
        super(TemporalCrossAttentionLayer, self).__init__()
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

        self.learned_pe = learned_pe
        self.pos_embed_x = nn.Parameter(torch.zeros(1, seq_len, d_model)) if learned_pe else PositionalEmbedding(d_model)
        self.pos_embed_cross = nn.Parameter(torch.zeros(1, pred_len, d_model)) if learned_pe else PositionalEmbedding(d_model)
        self.pos_drop = nn.Dropout(pos_drop)

        # weight initialization
        if learned_pe:
            nn.init.trunc_normal_(self.pos_embed_x, std=.02)
            nn.init.trunc_normal_(self.pos_embed_cross, std=.02)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        # if x.shape[0] == 0:
        #     return x, None

        B, T, H, W, D = queries.shape
        # rearrange and segment
        queries = rearrange(queries, 'b t h w d -> (b h w) t d')
        keys = rearrange(keys, 'b t h w d -> (b h w) t d')
        values = rearrange(values, 'b t h w d -> (b h w) t d')

        # positional embedding
        if self.learned_pe:
            queries = self.pos_drop(queries + self.pos_embed_x)
            keys = self.pos_drop(keys + self.pos_embed_cross)
            values = self.pos_drop(values + self.pos_embed_cross)
        else:
            queries = self.pos_drop(queries + self.pos_embed_x(queries))
            keys = self.pos_drop(keys + self.pos_embed_cross(keys))
            values = self.pos_drop(values + self.pos_embed_cross(values))

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
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x, attn


class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0.):
        super(TemporalSelfAttentionLayer, self).__init__()
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

        self.learned_pe = learned_pe
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model)) if learned_pe else PositionalEmbedding(d_model)
        self.pos_drop = nn.Dropout(pos_drop)

        # weight initialization
        if learned_pe:
            nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        # if x.shape[0] == 0:
        #     return x, None

        B, T, H, W, D = queries.shape
        # rearrange and segment
        queries = rearrange(queries, 'b t h w d -> (b h w) t d')
        keys = rearrange(keys, 'b t h w d -> (b h w) t d')
        values = rearrange(values, 'b t h w d -> (b h w) t d')

        # positional embedding
        if self.learned_pe:
            queries = self.pos_drop(queries + self.pos_embed)
            keys = self.pos_drop(keys + self.pos_embed)
            values = self.pos_drop(values + self.pos_embed)
        else:
            queries = self.pos_drop(queries + self.pos_embed(queries))
            keys = self.pos_drop(keys + self.pos_embed(keys))
            values = self.pos_drop(values + self.pos_embed(values))

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
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x, attn


class FourierLayer(nn.Module):
    def __init__(self, seq_len, top_k, d_model, img_size, fuse_drop=0.):
        super(FourierLayer, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_freqs = seq_len // 2
        self.height, self.width = img_size
        
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)
        # get the number of convolutional layers
        self.num_conv_layers, num_spatial_tokens = self.get_num_conv_layers()
        # convolutional layers
        self.Convs = nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.Convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2), 
                    nn.BatchNorm2d(d_model),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        self.fuse_proj = nn.Linear(d_model * num_spatial_tokens, 1)
        self.fuse_drop = nn.Dropout(fuse_drop)

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

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
        x = x.squeeze(-1)  # [B, T]

        # fft
        xf = torch.fft.rfft(x, dim=-1, norm='ortho')[:, 1:]  # [B, T//2]
        amp = torch.abs(xf) # [B, T//2]
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

        return gates  # [B, Ps]


class PeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, segment_size, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0.):
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
        x = rearrange(x, 'b t h w d -> (b h w) t d')
        padding_len = self.segment_size - self.seq_len % self.segment_size if self.seq_len % self.segment_size != 0 else 0
        x = F.pad(x, (0, 0, 0, padding_len), mode='replicate')
        x = x.unfold(1, self.segment_size, self.segment_size)
        x = rearrange(x, 'b n d p -> b n (p d)')
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

        # positional embedding
        if self.learned_pe:
            queries = self.pos_drop(queries + self.pos_embed)
            keys = self.pos_drop(keys + self.pos_embed)
            values = self.pos_drop(values + self.pos_embed)
        else:
            queries = self.pos_drop(queries + self.pos_embed(queries))
            keys = self.pos_drop(keys + self.pos_embed(keys))
            values = self.pos_drop(values + self.pos_embed(values))

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
        x = rearrange(x, 'b n (p d) -> b (n p) d', p=self.segment_size)[:,:self.seq_len]
        x = rearrange(x, '(b h w) t d -> b t h w d', h=H, w=W)

        return x, attn


class MultiScalePeriodicAttentionLayer(nn.Module):
    def __init__(self, attention, gate_layer, seq_len, d_model, n_heads, learned_pe=False, qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., pos_drop=0.):
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
                )
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # queries, keys, values [B, T, H, W, D]
        B, T, H, W, D = x.shape

        # gating
        gates = self.gate_layer(x, training=self.training) # [B, Ps]

        # dispatch
        xs = dispatch(x, gates) # Ps * [B, T, H, W, D]
        
        # multi-branch attention
        # attns = []
        # for i, (x_item, attention_layer) in enumerate(zip(xs, self.attention_layers)):
        #     x_item, attn = attention_layer(x_item, attn_mask=attn_mask, tau=tau, delta=delta)
        #     xs[i] = x_item
        #     attns.append(attn)
        xs = [attention_layer(x_item, x_item, x_item, attn_mask=attn_mask, tau=tau, delta=delta)[0] for x_item, attention_layer in zip(xs, self.attention_layers)]
        attns = None

        # combine
        x = combine(xs, gates)

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
        x = rearrange(x, '(b t) h w d -> b t h w d', t=T)

        return x, attn


class MultiScalePeriodicSpatialTemporalAttentionLayer(nn.Module):
    pass


class MSPSTTEncoderLayer(nn.Module):
    r"""

    Encoder layer for Multi-Scale Periodic Spatial-Temporal Transformer.
    Through swapping the order of the spatial attention (attention_s) and the temporal attention (attention_t), we can get two different versions of the encoder layer.

    """
    def __init__(self,
                 attention_t,
                 attention_s,
                 d_model: int,
                 d_ff: int = None,
                 dropout: float = 0.,
                 activation: str = "relu",
                 pre_norm: bool = False,
                 parallelize: bool = False):
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
        self.parallelize = parallelize # parallelize spatial attention and temporal attention

    def forward_parallel(self, x, attn_mask=None, tau=None, delta=None):
        res = x
        if self.pre_norm:
            x_t = self.norm_attn_t(x)
        else:
            x_t = x
        x_t, attn_t = self.attention_t(x_t)
        x_t = res + self.dropout(x_t)
        if not self.pre_norm:
            x_t = self.norm_attn_t(x_t)

        res = x_t
        if self.pre_norm:
            x_t = self.norm_mlp_t(x_t)
        x_t = self.mlp_t(x_t)
        x_t = res + self.dropout(x_t)
        if not self.pre_norm:
            x_t = self.norm_mlp_t(x_t)

        res = x
        if self.pre_norm:
            x_s = self.norm_attn_s(x)
        else:
            x_s = x
        x_s, attn_s = self.attention_s(x_s)
        x_s = res + self.dropout(x_s)
        if not self.pre_norm:
            x_s = self.norm_attn_s(x_s)
        
        res = x_s
        if self.pre_norm:
            x_s = self.norm_mlp_s(x_s)
        x_s = self.mlp_s(x_s)
        x_s = res + self.dropout(x_s)
        if not self.pre_norm:
            x_s = self.norm_mlp_s(x_s)

        ## combine
        # add
        x = (x_t + x_s) / 2
        # multiply
        # x = x_t * x_s
        # linear
        # x = self.linear(torch.cat([x_t, x_s], dim=-1))

        return x, (attn_t, attn_s)

    def forward_serial(self, x, attn_mask=None, tau=None, delta=None):
        res = x
        if self.pre_norm:
            x = self.norm_attn_t(x)
        x, attn_t = self.attention_t(x)
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

        res = x
        if self.pre_norm:
            x = self.norm_attn_s(x)
        x, attn_s = self.attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_attn_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_mlp_s(x)
        x = self.mlp_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp_s(x)

        return x, (attn_t, attn_s)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        # parallel
        if self.parallelize:
            return self.forward_parallel(x, attn_mask=attn_mask, tau=tau, delta=delta)

        # serial
        else:
            return self.forward_serial(x, attn_mask=attn_mask, tau=tau, delta=delta)


class MSPSTTEncoder(nn.Module):
    r"""
    
    Encoder for Multi-Scale Periodic Spatial-Temporal Transformer.
    
    """
    def __init__(self, 
                 encoder_layers,
                 norm_layer=None):
        super(MSPSTTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for encoder_layer in self.encoder_layers:
            x, attn = encoder_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class MSPSTTDecoderLayer(nn.Module):
    r"""
    
    Decoder layer for Multi-Scale Periodic Spatial-Temporal Transformer.
    Through swapping the order of the spatial attention (attention_s) and the temporal attention (attention_t), we can get two different versions of the decoder layer.
    
    """
    def __init__(self, 
                self_attention_t,
                self_attention_s,
                cross_attention_t,
                cross_attention_s,
                d_model: int,
                d_ff: int = None,
                dropout: float = 0.,
                activation: str = "relu",
                pre_norm: bool = False,
                parallelize: bool = False):
        super(MSPSTTDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        activation = nn.ReLU if activation == "relu" else nn.GELU
        
        self.self_attention_t = self_attention_t
        self.self_attention_s = self_attention_s
        self.cross_attention_t = cross_attention_t
        self.cross_attention_s = cross_attention_s
        self.mlp = Mlp(in_features=d_model, hidden_features=d_ff, act_layer=activation, drop=dropout)
        self.norm_sa_t = nn.LayerNorm(d_model)
        self.norm_sa_s = nn.LayerNorm(d_model)
        self.norm_ca_t = nn.LayerNorm(d_model)
        self.norm_ca_s = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.pre_norm = pre_norm
        self.parallelize = parallelize # parallelize spatial attention and temporal attention

    def forward_parallel(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        res = x
        if self.pre_norm:
            x_t = self.norm_sa_t(x)
        else:
            x_t = x
        x_t, self_attn_t = self.self_attention_t(x_t, x_t, x_t) # Temporal Self-Attention
        x_t = res + self.dropout(x_t)
        if not self.pre_norm:
            x_t = self.norm_sa_t(x_t)

        res = x
        if self.pre_norm:
            x_s = self.norm_sa_s(x)
        else:
            x_s = x
        x_s, self_attn_s = self.self_attention_s(x_s)
        x_s = res + self.dropout(x_s)
        if not self.pre_norm:
            x_s = self.norm_sa_s(x_s)
        
        ## combine
        # add
        x = (x_t + x_s) / 2
        # multiply
        # x = x_t * x_s
        # linear
        # x = self.self_reduction(torch.cat([x_t, x_s], dim=-1))

        res = x
        if self.pre_norm:
            x = self.norm_ca_t(x)
            cross = self.norm_cross(cross)
        x, cross_attn_t = self.cross_attention_t(x, cross, cross) # Temporal Cross-Attention
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_ca_t(x)

        res = x
        if self.pre_norm:
            x_s = self.norm_ca_s(x)
        x, cross_attn_s = self.cross_attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_ca_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_mlp(x)
        x = self.mlp(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp(x)

        return x

    def forward_serial(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        res = x
        if self.pre_norm:
            x = self.norm_sa_t(x)
        x, self_attn_t = self.self_attention_t(x, x, x) # Temporal Self-Attention
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_sa_t(x)

        res = x
        if self.pre_norm:
            x = self.norm_sa_s(x)
        x, self_attn_s = self.self_attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_sa_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_ca_t(x)
            cross = self.norm_cross(cross)
        x, cross_attn_t = self.cross_attention_t(x, cross, cross) # Temporal Cross-Attention
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_ca_t(x)

        res = x
        if self.pre_norm:
            x = self.norm_ca_s(x)
        x, cross_attn_s = self.cross_attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_ca_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_mlp(x)
        x = self.mlp(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp(x)

        return x

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x [B, T, H, W, D]
        # parallel
        if self.parallelize:
            return self.forward_parallel(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # serial
        else:
            return self.forward_serial(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)


class MSPSTTDecoder(nn.Module):
    r"""
    
    Decoder for Multi-Scale Periodic Spatial-Temporal Transformer.
    
    """
    def __init__(self, 
                 decoder_layers,
                 norm_layer=None,
                 projection=None):
        super(MSPSTTDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
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
    r"""Patch Recovery Module [B, T, H, W, D] -> [B, T, H, W, C]"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchRecovery, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b t h w c', t=T)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = F.pad(x, pad_values)
        _, H, W, _ = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.
    Args:
        dim: Number of input channels.
        out_dim: Number of output channels (or 2 * dim if None)
        norm_layer: Normalization layer.
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self,
                 img_size: tuple,
                 patch_size: int,
                 in_chans: int = 3,
                 d_model: int = 512,
                 embed_type: str = 'fixed',
                 freq: str = 'h',
                 dropout: float = 0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type,
            freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) # B T H W D
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark).unsqueeze(-2).unsqueeze(-2)
        return self.dropout(x)



class Model(nn.Module):
    r'''
    
    Multi-Scale Periodic Spatio-Temporal Transformer ------ Encoder-Decoder Architecture
    
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
        assert self.height % self.patch_size == 0, "Height must be divisible by patch size"
        assert self.width % self.patch_size == 0, "Width must be divisible by patch size"

        self.enc_embedding = DataEmbedding(img_size=self.img_size, patch_size=configs.patch_size, in_chans=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)

        self.dec_embedding = DataEmbedding(img_size=self.img_size, patch_size=configs.patch_size, in_chans=configs.dec_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)

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
                                FourierLayer(
                                    seq_len=configs.seq_len,
                                    top_k=configs.top_k,
                                    d_model=configs.d_model,
                                    img_size=(configs.height//self.patch_size, configs.width//self.patch_size),
                                    fuse_drop=configs.dropout,
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
                                input_resolution=(configs.height//self.patch_size, configs.width//self.patch_size),
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size//2,
                                always_partition=False,
                                dynamic_mask=False,
                            ),
                            d_model=configs.d_model,
                            d_ff=configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                            pre_norm=configs.pre_norm,
                            parallelize=configs.is_parallel,
                        )
                        for i in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )

        self.decoder = MSPSTTDecoder(
                    [
                        MSPSTTDecoderLayer(
                            TemporalSelfAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=True,
                                    output_attention=configs.output_attention,
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
                                input_resolution=(configs.height//self.patch_size, configs.width//self.patch_size),
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size//2,
                                always_partition=False,
                                dynamic_mask=False,
                            ),
                            TemporalCrossAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                seq_len=configs.seq_len,
                                pred_len=configs.pred_len,
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                learned_pe=False,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                pos_drop=configs.dropout,
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
                                input_resolution=(configs.height//self.patch_size, configs.width//self.patch_size),
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size//2,
                                always_partition=False,
                                dynamic_mask=False,
                            ),
                            d_model=configs.d_model,
                            d_ff=configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                            pre_norm=configs.pre_norm,
                            parallelize=configs.is_parallel,
                        )
                        for i in range(configs.d_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model),
                    projection=PatchRecovery(self.img_size, self.patch_size, configs.c_out, configs.d_model)
                )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        # x_enc [B, T, H, W, C]

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # -> [B, T, H', W', D]
        dec_out = self.dec_embedding(x_dec, x_mark_dec) # -> [B, S, H', W', D]
        
        # encoding
        enc_out, _ = self.encoder(enc_out) # -> [B, T, H', W', D]

        # decoding
        dec_out = self.decoder(dec_out, enc_out) # -> [B, S, H', W', D]

        return dec_out, None