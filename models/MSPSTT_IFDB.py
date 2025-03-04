from typing import List, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import log, ceil
from einops import rearrange
from timm.layers import Mlp, LayerNorm2d, to_2tuple
from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding
from layers.MSPSTT_Modules import MultiScalePeriodicAttentionLayer, VisionTransformerAttentionLayer, TemporalAttentionLayer, FullAttention, precompute_freqs_cis, apply_rotary_emb


class MSPSTTEncoderLayer(nn.Module):
    r"""

    Encoder layer for Multi-Scale Periodic Spatial-Temporal Transformer.

    """
    def __init__(
            self,
            attention_t,
            attention_s,
            d_model: int,
            d_ff: int = None,
            dropout: float = 0.,
            activation: str = "relu",
            pre_norm: bool = False,
            parallelize: bool = False
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

        self.parallel_norm_t = nn.LayerNorm(d_model) if parallelize else None
        self.parallel_norm_s = nn.LayerNorm(d_model) if parallelize else None
        self.concat = nn.Linear(2*d_model, d_model) if parallelize else None
        
        self.pre_norm = pre_norm # pre-norm or post-norm
        self.parallelize = parallelize # parallelize spatial attention and temporal attention

    def forward_parallel(self, x, freq_cis=None, attn_mask=None):
        res = x
        if self.pre_norm:
            x_t = self.norm_attn_t(x)
        else:
            x_t = x
        x_t, balance_loss = self.attention_t(x_t, freq_cis, attn_mask)
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
        x_s = self.attention_s(x_s)
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
        x_t = self.parallel_norm_t(x_t)
        x_s = self.parallel_norm_s(x_s)
        x = self.concat(torch.cat([x_t + x_s, x_t * x_s], dim=-1))

        return x, balance_loss

    def forward_serial(self, x, freq_cis=None, attn_mask=None):
        res = x
        if self.pre_norm:
            x = self.norm_attn_t(x)
        x, balance_loss = self.attention_t(x, freq_cis, attn_mask)
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
        x = self.attention_s(x)
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

        return x, balance_loss

    def forward(self, x, freq_cis=None, attn_mask=None):
        # x [B, T, H, W, D]
        return self.forward_parallel(x, freq_cis, attn_mask=attn_mask) if self.parallelize else self.forward_serial(x, freq_cis, attn_mask=attn_mask)


class MSPSTTEncoder(nn.Module):
    r"""
    
    Encoder for Multi-Scale Periodic Spatial-Temporal Transformer.
    
    """
    def __init__(
            self, 
            encoder_layers: List[nn.Module],
            norm_layer: nn.Module = None
    ):
        super(MSPSTTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, freq_cis=None, attn_mask=None):
        balance_loss = 0.
        for encoder_layer in self.encoder_layers:
            x, aux_loss = encoder_layer(x, freq_cis, attn_mask)
            balance_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, balance_loss


class MSPSTTDecoderLayer(nn.Module):
    r"""
    
    Decoder layer for Multi-Scale Periodic Spatial-Temporal Transformer.
    
    """
    def __init__(
            self, 
            self_attention_t,
            self_attention_s,
            cross_attention_t,
            cross_attention_s,
            d_model: int,
            d_ff: int = None,
            dropout: float = 0.,
            activation: str = "relu",
            pre_norm: bool = False,
            parallelize: bool = False
    ):
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

        self.parallel_norm_sa_t = nn.LayerNorm(d_model) if parallelize else None
        self.parallel_norm_sa_s = nn.LayerNorm(d_model) if parallelize else None
        self.parallel_norm_ca_t = nn.LayerNorm(d_model) if parallelize else None
        self.parallel_norm_ca_s = nn.LayerNorm(d_model) if parallelize else None
        self.concat_sa = nn.Linear(2*d_model, d_model) if parallelize else None
        self.concat_ca = nn.Linear(2*d_model, d_model) if parallelize else None
        
        self.pre_norm = pre_norm # pre-norm or post-norm
        self.parallelize = parallelize # parallelize spatial attention and temporal attention

    def forward_parallel(self, x, cross, freq_cis=None, x_mask=None, cross_mask=None):
        res = x
        if self.pre_norm:
            x_t = self.norm_sa_t(x)
        else:
            x_t = x
        x_t = self.self_attention_t(x_t, x_t, x_t, freq_cis, x_mask) # Temporal Self-Attention
        x_t = res + self.dropout(x_t)
        if not self.pre_norm:
            x_t = self.norm_sa_t(x_t)

        res = x
        if self.pre_norm:
            x_s = self.norm_sa_s(x)
        else:
            x_s = x
        x_s = self.self_attention_s(x_s)
        x_s = res + self.dropout(x_s)
        if not self.pre_norm:
            x_s = self.norm_sa_s(x_s)
        
        ## combine
        x_t = self.parallel_norm_sa_t(x_t)
        x_s = self.parallel_norm_sa_s(x_s)
        x = self.concat_sa(torch.cat([x_t + x_s, x_t * x_s], dim=-1))

        res = x
        if self.pre_norm:
            x_t = self.norm_ca_t(x)
            cross = self.norm_cross(cross)
        x_t = self.cross_attention_t(x_t, cross, cross, freq_cis, cross_mask) # Temporal Cross-Attention
        x_t = res + self.dropout(x_t)
        if not self.pre_norm:
            x_t = self.norm_ca_t(x_t)

        res = x
        if self.pre_norm:
            x_s = self.norm_ca_s(x)
        x_s = self.cross_attention_s(x_s)
        x_s = res + self.dropout(x_s)
        if not self.pre_norm:
            x_s = self.norm_ca_s(x_s)

        ## combine
        x_t = self.parallel_norm_ca_t(x_t)
        x_s = self.parallel_norm_ca_s(x_s)
        x = self.concat_ca(torch.cat([x_t + x_s, x_t * x_s], dim=-1))

        res = x
        if self.pre_norm:
            x = self.norm_mlp(x)
        x = self.mlp(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_mlp(x)

        return x

    def forward_serial(self, x, cross, freq_cis=None, x_mask=None, cross_mask=None):
        res = x
        if self.pre_norm:
            x = self.norm_sa_t(x)
        x = self.self_attention_t(x, x, x, freq_cis, x_mask) # Temporal Self-Attention
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_sa_t(x)

        res = x
        if self.pre_norm:
            x = self.norm_sa_s(x)
        x = self.self_attention_s(x)
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_sa_s(x)

        res = x
        if self.pre_norm:
            x = self.norm_ca_t(x)
            cross = self.norm_cross(cross)
        x = self.cross_attention_t(x, cross, cross, freq_cis, cross_mask) # Temporal Cross-Attention
        x = res + self.dropout(x)
        if not self.pre_norm:
            x = self.norm_ca_t(x)

        res = x
        if self.pre_norm:
            x = self.norm_ca_s(x)
        x = self.cross_attention_s(x)
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

    def forward(self, x, cross, freq_cis=None, x_mask=None, cross_mask=None):
        # x [B, T, H, W, D]
        return self.forward_parallel(x, cross, freq_cis, x_mask=x_mask, cross_mask=cross_mask) if self.parallelize else self.forward_serial(x, cross, freq_cis, x_mask=x_mask, cross_mask=cross_mask)


class MSPSTTDecoder(nn.Module):
    r"""
    
    Decoder for Multi-Scale Periodic Spatial-Temporal Transformer.
    
    """
    def __init__(
            self, 
            decoder_layers: List[nn.Module],
            norm_layer: nn.Module = None,
            projection: nn.Module = None,
        ):
        super(MSPSTTDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, freq_cis=None, x_mask=None, cross_mask=None):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, cross, freq_cis, x_mask=x_mask, cross_mask=cross_mask)

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


class PatchRecovery_OneStep(nn.Module):
    r"""Patch recovery module. 
    
    Args:
        patch_size (int): Size of the patch.
        embed_dim (int): Number of linear projection output channels.
        out_chans (int): Number of output channels.
    """

    def __init__(
            self, 
            patch_size: int = 16,
            embed_dim: int = 512,
            out_chans: int = 3,
    ):
        super(PatchRecovery_OneStep, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.proj = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_size)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b t h w c', t=T)
        return x


class PatchRecovery_StepByStep(nn.Module):
    r""" Patch recovery step by step.

    Args:
        patch_size (int): Size of the patch.
        embed_dim (int): Number of linear projection output channels.
        out_chans (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the transposed convolution.
        padding (int): Padding of the transposed convolution.
        output_padding (int): Output padding of the transposed convolution.
    """

    def __init__(
            self,
            patch_size: int = 16,
            embed_dim: int = 512,
            out_chans: int = 3,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
            output_padding: int = 1
    ):
        super(PatchRecovery_StepByStep, self).__init__()

        num_layers = np.log2(patch_size).astype(int) - 1
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)

        self.Convs = nn.ModuleList()
        for i in range(num_layers):
            self.Convs.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=embed_dim,out_channels=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
                LayerNorm2d(embed_dim),
                nn.SiLU(inplace=True)
            ))
        self.Convs.append(nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        for conv_layer in self.Convs:
            x = conv_layer(x)
        x = rearrange(x, '(b t) c h w -> b t h w c', t=T)
        return x


class PatchRecovery_PixelShuffle(nn.Module):
    r""" Patch recovery with pixel shuffle.

    Args:
        patch_size (int): Size of the patch.
        embed_dim (int): Number of linear projection output channels.
        out_chans (int): Number of output channels.
    """

    def __init__(
            self,
            patch_size: int = 16,
            embed_dim: int = 512,
            out_chans: int = 3,
    ):
        super(PatchRecovery_PixelShuffle, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.proj = nn.Conv2d(in_channels=embed_dim, out_channels=out_chans * patch_size ** 2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self, x):
        # x [B, T, H, W, D]
        B, T, H, W, D = x.shape
        x = rearrange(x, 'b t h w d -> (b t) d h w')
        x = self.proj(x)
        x = self.pixel_shuffle(x)
        x = rearrange(x, '(b t) c h w -> b t h w c', t=T)
        return x


class DataEmbedding_wo_pos(nn.Module):
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
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark).unsqueeze(-2).unsqueeze(-2)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEmbedding2D(torch.nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEmbedding2D, self).__init__()
        self.d_model = d_model

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))

        pe_2d = torch.zeros(height, width, d_model)
        pe_2d.require_grad = False
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(log(10000.0) / d_model))  # [d_model/2]

        pos_w = torch.arange(0., width).unsqueeze(1)  # [W, 1]
        pos_h = torch.arange(0., height).unsqueeze(1)  # [H, 1]

        pe_2d[:, :, 0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe_2d[:, :, 1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe_2d[:, :, d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
        pe_2d[:, :, d_model + 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)

        pe_2d = pe_2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe_2d', pe_2d)

    def forward(self, x):
        return self.pe_2d[:, :, :x.size(2), :x.size(3), :]


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
        self.temporal_position_embedding = PositionalEmbedding(d_model=d_model)
        self.spatial_position_embedding = PositionalEmbedding2D(d_model=d_model, height=img_size[0]//patch_size, width=img_size[1]//patch_size)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.temporal_position_embedding(x) + self.spatial_position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_position_embedding(x) + self.spatial_position_embedding(x) + self.temporal_embedding(x_mark).unsqueeze(-2).unsqueeze(-2)
        return self.dropout(x)


class Model(nn.Module):
    r'''
    
    Multi-Scale Periodic Spatio-Temporal Transformer ------ Informer Encoder-Decoder Architecture
    
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
        self.patch_size = configs.patch_size
        assert self.height % self.patch_size == 0, "Height must be divisible by patch size"
        assert self.width % self.patch_size == 0, "Width must be divisible by patch size"

        self.enc_embedding = DataEmbedding_wo_pos(img_size=(configs.height, configs.width), patch_size=configs.patch_size, in_chans=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout) if configs.is_rotary else DataEmbedding(img_size=(configs.height, configs.width), patch_size=configs.patch_size, in_chans=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)

        self.dec_embedding = DataEmbedding_wo_pos(img_size=(configs.height, configs.width),patch_size=configs.patch_size, in_chans=configs.dec_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout) if configs.is_rotary else DataEmbedding(img_size=(configs.height, configs.width), patch_size=configs.patch_size, in_chans=configs.dec_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout)

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
                                img_size=(configs.height//self.patch_size, configs.width//self.patch_size),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                use_conv=configs.use_conv,
                                use_linear=configs.use_linear,
                                is_rotary=configs.is_rotary,
                                dropout=configs.dropout,
                            ),
                            VisionTransformerAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                attn_drop=configs.dropout,
                                proj_drop=configs.dropout,
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
                            TemporalAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=True,
                                    output_attention=configs.output_attention,
                                ),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                is_rotary=configs.is_rotary,
                            ),
                            VisionTransformerAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                attn_drop=configs.dropout,
                                proj_drop=configs.dropout,
                            ),
                            TemporalAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                qk_norm=False,
                                proj_bias=True,
                                proj_drop=configs.dropout,
                                is_rotary=configs.is_rotary,
                            ),
                            VisionTransformerAttentionLayer(
                                FullAttention(
                                    d_model=configs.d_model,
                                    n_heads=configs.n_heads,
                                    attn_drop=configs.dropout,
                                    is_causal=False,
                                    output_attention=configs.output_attention,
                                ),
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                qkv_bias=True,
                                attn_drop=configs.dropout,
                                proj_drop=configs.dropout,
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
                    projection=PatchRecovery_StepByStep(patch_size=self.patch_size, embed_dim=configs.d_model, out_chans=configs.c_out)
                )

        # rotary position encoding for encoder
        self.segment_sizes = self.get_segment_sizes(self.seq_len)
        d_scale = np.mean(self.segment_sizes).astype(int)
        self.register_buffer("freqs_cis_enc", precompute_freqs_cis(d_scale*configs.d_model, self.seq_len), persistent=False)

        # rotary position encoding for decoder
        self.register_buffer("freqs_cis_dec", precompute_freqs_cis(configs.d_model, max([configs.seq_len,  configs.label_len + configs.pred_len])), persistent=False)

    def get_segment_sizes(self, seq_len):
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        segment_sizes = (peroid_list + 1e-5).int().unique().detach().cpu().numpy()[::-1]
        print(f"Segment sizes: {segment_sizes}")
        return segment_sizes

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_true=None):
        # x_enc [B, T, H, W, C]
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # -> [B, T, H', W', D]
        dec_out = self.dec_embedding(x_dec, x_mark_dec) # -> [B, S, H', W', D]
        # encoder
        freqs_cis_enc = getattr(self, "freqs_cis_enc", None)
        enc_out, balance_loss = self.encoder(enc_out, freqs_cis_enc)
        # decoder
        freqs_cis_dec = getattr(self, "freqs_cis_dec", None)
        dec_out = self.decoder(dec_out, enc_out, freqs_cis_dec) # -> [B, S, H', W', D]
        return dec_out, balance_loss