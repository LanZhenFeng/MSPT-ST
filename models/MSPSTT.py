import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import ceil
from einops import rearrange

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
from layers.Embed import PositionalEmbedding2D
from layers.SelfAttention_Family import FullAttention, AttentionLayer


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


class MLP(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


class MSPTSTEncoderLayer(nn.Module):
    def __init__(self, attention_v, attention_t, attention_s, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(MSPTSTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_v = attention_v
        self.attention_t = attention_t
        self.attention_s = attention_s
        self.mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, S, D]
        B, T, S, D = x.shape
        x = rearrange(x, 'b t s d -> (b t) s d')
        # Spatio Attention
        res = x
        x, attn = self.attention_s(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm1(res + self.dropout(x))

        x = rearrange(x, '(b t) s d -> (b s) t d', t=T)

        # Temporal Attention
        res = x
        x, attn = self.attention_t(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = self.norm2(res + self.dropout(x))

        x = rearrange(x, '(b s) t d -> b t s d', s=S)

        # MLP
        res = x
        x = self.mlp(x)
        x = self.norm3(res + self.dropout(x))

        return x, attn

class MSPTSTEncoderLayer_PreNorm(nn.Module):
    def __init__(self, attention_v, attention_t, attention_s, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(MSPTSTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_v = attention_v
        self.attention_t = attention_t
        self.attention_s = attention_s
        self.mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, T, S, D]
        B, T, S, D = x.shape
        x = rearrange(x, 'b t s d -> (b t) s d')
        # Spatio Attention
        res = x
        x = self.norm1(x)
        x, attn = self.attention_s(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = res + self.dropout(x)

        x = rearrange(x, '(b t) s d -> (b s) t d', t=T)

        # Temporal Attention
        res = x
        x = self.norm2(x)
        x, attn = self.attention_t(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = res + self.dropout(x)

        x = rearrange(x, '(b s) t d -> b t s d', s=S)

        # MLP
        res = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = res + self.dropout(x)

        return x, attn



class MSPTSTEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(MSPTSTEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)

class FourierLayer(nn.Module):
    def __init__(self, seq_len, top_k, in_chans, height, width, dropout=0.1):
        super(FourierLayer, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.num_freqs = seq_len // 2
        # get the segment sizes
        self.segment_sizes = self.get_segment_sizes(seq_len)
        self.embed_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=in_chans*4, kernel_size=4, stride=4), # [BT, C*4, H//4, W//4]
            Permute(0, 2, 3, 1), # [BT, H//4, W//4, C*4]
            nn.LayerNorm(in_chans*4),
            nn.GELU(),
            Permute(0, 3, 1, 2), # [BT, C*4, H//4, W//4]
            nn.Conv2d(in_channels=in_chans*4, out_channels=in_chans*16, kernel_size=4, stride=4), # [BT, C*16, H//16, W//16]
            Permute(0, 2, 3, 1), # [BT, H//16, W//16, C*16]
            nn.LayerNorm(in_chans*16),
            nn.GELU(),
            nn.Flatten(1, -1), # [BT, C*16*H//16*W//16]
            nn.Linear(in_features=in_chans*16*height//16*width//16, out_features=512)
        ) 

        # Noise parameters
        self.w_gate = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))
        self.w_noise = nn.Parameter(torch.zeros(self.num_freqs, len(self.segment_sizes)))

        self.dropout = nn.Dropout(dropout)

    def get_segment_sizes(self, seq_len):
        # get the period list, first element is inf
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
        # segment_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        print(f"Segment sizes: {segment_sizes}")
        return segment_sizes

    def forward(self, x, training=False, noise_epsilon=1e-2):
        # x [B, T, H, W, C]
        B, T, H, W, C = x.shape
        # fourier embed
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.embed_layer(x) # [BT, 512]
        x = self.dropout(x)
        x = rearrange(x, '(b t) d -> b d t', b=B)

        # fft
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

class CubeEmbed(nn.Module):
    def __init__(self, seq_len, img_size, segment_size, patch_size, stride_t, stride_s, padding_t, padding_s, in_chans, embed_dim):
        super(CubeEmbed, self).__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.segment_size = segment_size
        self.patch_size = patch_size
        self.stride_t = stride_t
        self.stride_s = stride_s
        self.padding_t = padding_t
        self.padding_s = padding_s
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(segment_size, patch_size, patch_size), stride=(stride_t, stride_s, stride_s), padding=(padding_t, padding_s, padding_s))
        self.norm = nn.LayerNorm(embed_dim)
        self.num_cubes = ((seq_len - segment_size) // stride_t + 1) * (img_size[0] // stride_s) * (img_size[1] // stride_s)

    def forward(self, x):
        # x [B, C, T, H, W]
        x = self.proj(x) # -> [B, D, T//S, H//P, W//P]
        x = rearrange(x, 'b d t h w -> b t (h w) d') # -> [B, T, H*W, D]
        x = self.norm(x)
        return x

class MultiScaleCubeEmbed(nn.Module):
    def __init__(self, seq_len, segment_sizes, img_size, patch_size, in_chans, embed_dim, dropout=0., stride_scale=1):
        super(MultiScaleCubeEmbed, self).__init__()
        self.seq_len = seq_len
        self.segment_sizes = segment_sizes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        assert stride_scale == 1 or stride_scale == 2, "stride_scale must be 1 or 2"
        self.embed_layers = nn.ModuleList()
        padding_t = 0
        stride_s = patch_size // stride_scale
        padding_s = stride_s // 2 if stride_scale == 2 else 0
        for segment_size in segment_sizes:
            stride_t = ceil(segment_size / stride_scale)
            self.embed_layers.append(CubeEmbed(seq_len, img_size, segment_size, patch_size, stride_t, stride_s, padding_t, padding_s, in_chans, embed_dim))
        H, W = img_size[0] // stride_s, img_size[1] // stride_s
        self.positional_encoding = PositionalEmbedding2D(embed_dim, seq_len+1, H*W)
        self.pos_drop = nn.Dropout(dropout)

    def forward(self, x, gates):
        # x [B, T, H, W, C]
        x = rearrange(x, 'b t h w c -> b c t h w') # -> [B, C, T, H, W]
        xs = dispatch(x, gates) # -> Ps * [B, C, T, H, W]
        xs = [embed(F.pad(x, (0, 0, 0, 0, 0, segment_size - self.seq_len % segment_size), mode='replicate')) for x, segment_size, embed in zip(xs, self.segment_sizes, self.embed_layers)]
        xs = [self.pos_drop(x + self.positional_encoding(x)) for x in xs]
        return xs

class CubeRecovery(nn.Module):
    def __init__(self, seq_len, patch_size, stride_s, padding_s, d_model, c_out, dropout=0.):
        super(CubeRecovery, self).__init__()
        self.proj = nn.ConvTranspose3d(d_model, c_out, kernel_size=(seq_len, patch_size, patch_size), stride=(seq_len, stride_s, stride_s), padding=(0, padding_s, padding_s))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [B, T, H, W, D]
        x = rearrange(x, 'b t h w d -> b d t h w') # -> [B, D, T, H, W]
        x = self.proj(self.dropout(x)) # -> [B, C, T, H, W]
        x = rearrange(x, 'b c t h w -> b t h w c')
        return x

class CrossScaleAggregator(nn.Module):
    def __init__(self, seq_len, img_size, segment_sizes, patch_size, d_model, c_out, dropout=0., stride_scale=1):
        super(CrossScaleAggregator, self).__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.height = img_size[0]
        self.width = img_size[1]
        self.segment_sizes = segment_sizes
        self.patch_size = patch_size
        self.d_model = d_model
        self.c_out = c_out
        self.stride_scale = stride_scale
        assert stride_scale == 1 or stride_scale == 2, "stride_scale must be 1 or 2"
        padding_t = 0
        stride_s = patch_size // stride_scale
        padding_s = stride_s // 2 if stride_scale == 2 else 0
        self.H, self.W = img_size[0] // stride_s, img_size[1] // stride_s
        self.reconvery_t_layers = nn.ModuleList()
        for segment_size in segment_sizes:
            stride_t = ceil(segment_size / stride_scale)
            self.reconvery_t_layers.append(nn.ConvTranspose3d(d_model, d_model, kernel_size=(segment_size, 3, 3), stride=(stride_t, 1, 1), padding=(padding_t, 1, 1)))
        
        self.reconvery_s = CubeRecovery(seq_len, patch_size, stride_s, padding_s, d_model, c_out, dropout)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, gates):
        # xs Ps * [B, T, S, D]
        xs = [rearrange(x, 'b t (h w) d -> b d t h w', h=self.H, w=self.W) for x in xs] # -> Ps * [B, D, T, H, W]
        xs = [recovery_t(self.dropout(x))[:, :, :self.seq_len] for x, recovery_t in zip(xs, self.reconvery_t_layers)] # -> Ps * [B, D, T, H, W]
        xs = combine(xs, gates) # -> [B, D, T, H, W]
        xs = rearrange(xs, 'b d t h w -> b t h w d') # -> [B, T, H, W, D]
        xs = self.norm(self.activation(xs))
        xs = self.reconvery_s(xs) # -> [B, T, H, W, C]
        return xs


class Model(nn.Module):
    def __init__(self, configs): 
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

        self.fourier_layer = FourierLayer(self.seq_len, configs.top_k, configs.enc_in, self.height, self.width, configs.dropout)
        self.segment_sizes = self.fourier_layer.segment_sizes

        self.multi_scale_cube_embed = MultiScaleCubeEmbed(self.seq_len, self.segment_sizes, self.img_size, self.patch_size, configs.enc_in, configs.d_model, configs.dropout, configs.stride_scale)

        self.encoders = nn.ModuleList()
        for segment_size in self.segment_sizes:
            self.encoders.append(
                MSPTSTEncoder(
                    [
                        MSPTSTEncoderLayer(
                            AttentionLayer(
                                FullAttention(False, attention_dropout=configs.dropout),
                                configs.d_model,
                                configs.n_heads,
                            ),
                            AttentionLayer(
                                FullAttention(False, attention_dropout=configs.dropout),
                                configs.d_model,
                                configs.n_heads,
                            ),
                            AttentionLayer(
                                FullAttention(False, attention_dropout=configs.dropout),
                                configs.d_model,
                                configs.n_heads,
                            ),
                            configs.d_model,
                            configs.d_ff,
                            configs.dropout,
                            configs.activation
                        )
                        for _ in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model)
                )
            )

        self.patch_recovery = CrossScaleAggregator(self.seq_len, self.img_size, self.segment_sizes, self.patch_size, configs.d_model, configs.enc_in, configs.dropout, configs.stride_scale)

    def forward_core(self, x_enc):
        # x_enc [B, T, H, W, C]
        
        # fourier layer
        gates = self.fourier_layer(x_enc)

        # multi-scale cube embedding
        xs = self.multi_scale_cube_embed(x_enc, gates)

        # encoding
        xs = [encoder(x)[0] for x, encoder in zip(xs, self.encoders)]

        # multi-scale cube recovery
        dec_out = self.patch_recovery(xs, gates)

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

if __name__ == "__main__":
    x = torch.randn(2, 3, 3, 2, 2, 4)
    gates = torch.zeros(2, 5)
    gates[0, 0] = 0.8
    gates[0, 1] = 0.2
    gates[1, 4] = 0.6
    gates[1, 3] = 0.4
    xs = dispatch(x, gates)
    combine(xs, gates)