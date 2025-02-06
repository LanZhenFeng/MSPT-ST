import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.Embed import PositionalEmbedding2D
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class GatedTransformerBlock(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation=nn.SiLU):
        super(GatedTransformerBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Attention
        res = x
        x = self.norm1(x)
        x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = res + self.dropout(x)

        # GLU
        res = x
        x = self.norm2(x)
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))) * self.conv2(x.transpose(-1, 1)))
        x = self.dropout(self.conv3(x).transpose(-1, 1))
        x = res + x

        return x, attn

class FullAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(FullAttentionLayer, self).__init__()
        self.gtb = GatedTransformerBlock(
            attention=attention,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> b (t s) d')
        x, attn = self.gtb(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, 'b (t s) d -> b t s d', t=T, s=S)
        return x, attn

class BinaryTSLayer(nn.Module):
    def __init__(self, attention_t, attention_s, d_model, d_ff=None, dropout=0.1):
        super(BinaryTSLayer, self).__init__()
        self.gtb_t = GatedTransformerBlock(
            attention=attention_t,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s = GatedTransformerBlock(
            attention=attention_s,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b s) t d')
        x, attn = self.gtb_t(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> (b t) s d', t=T, s=S)
        x, attn = self.gtb_s(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> b t s d', t=T, s=S)
        return x, None

class BinarySTLayer(nn.Module):
    def __init__(self, attention_s, attention_t, d_model, d_ff=None, dropout=0.1):
        super(BinaryTSLayer, self).__init__()
        self.gtb_t = GatedTransformerBlock(
            attention=attention_t,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s = GatedTransformerBlock(
            attention=attention_s,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b t) s d')
        x, attn = self.gtb_s(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> (b s) t d', t=T, s=S)
        x, attn = self.gtb_t(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> b t s d', t=T, s=S)
        return x, None

class TripletTSTLayer(nn.Module):
    def __init__(self, attention_t1, attention_s, attention_t2, d_model, d_ff=None, dropout=0.1):
        super(TripletTSTLayer, self).__init__()
        self.gtb_t1 = GatedTransformerBlock(
            attention=attention_t1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s = GatedTransformerBlock(
            attention=attention_s,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_t2 = GatedTransformerBlock(
            attention=attention_t2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b s) t d')
        x, attn = self.gtb_t1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> (b t) s d', t=T, s=S)
        x, attn = self.gtb_s(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> (b s) t d', t=T, s=S)
        x, attn = self.gtb_t2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> b t s d', t=T, s=S)
        return x, None

class TripletSTSLayer(nn.Module):
    def __init__(self, attention_s1, attention_t, attention_s2, d_model, d_ff=None, dropout=0.1):
        super(TripletSTSLayer, self).__init__()
        self.gtb_s1 = GatedTransformerBlock(
            attention=attention_s1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_t = GatedTransformerBlock(
            attention=attention_t,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s2 = GatedTransformerBlock(
            attention=attention_s2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b t) s d')
        x, attn = self.gtb_s1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> (b s) t d', t=T, s=S)
        x, attn = self.gtb_t(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> (b t) s d', t=T, s=S)
        x, attn = self.gtb_s2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> b t s d', t=T, s=S)
        return x, None

class QuadrupletTSSTLayer(nn.Module):
    def __init__(self, attention_t1, attention_s1, attention_s2, attention_t2, d_model, d_ff=None, dropout=0.1):
        super(QuadrupletTSSTLayer, self).__init__()
        self.gtb_t1 = GatedTransformerBlock(
            attention=attention_t1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s1 = GatedTransformerBlock(
            attention=attention_s1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s2 = GatedTransformerBlock(
            attention=attention_s2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_t2 = GatedTransformerBlock(
            attention=attention_t2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b s) t d')
        x, attn = self.gtb_t1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> (b t) s d', t=T, s=S)
        x, attn = self.gtb_s1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x, attn = self.gtb_s2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> (b s) t d', t=T, s=S)
        x, attn = self.gtb_t2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> b t s d', t=T, s=S)
        return x, None

class QuadrupletSTTSLayer(nn.Module):
    def __init__(self, attention_s1, attention_t1, attention_t2, attention_s2, d_model, d_ff=None, dropout=0.1):
        super(QuadrupletSTTSLayer, self).__init__()
        self.gtb_s1 = GatedTransformerBlock(
            attention=attention_s1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_t1 = GatedTransformerBlock(
            attention=attention_t1,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_t2 = GatedTransformerBlock(
            attention=attention_t2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s2 = GatedTransformerBlock(
            attention=attention_s2,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        T, S = x.shape[1], x.shape[2]
        x = rearrange(x, 'b t s d -> (b t) s d')
        x, attn = self.gtb_s1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> (b s) t d', t=T, s=S)
        x, attn = self.gtb_t1(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x, attn = self.gtb_t2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b s) t d -> (b t) s d', t=T, s=S)
        x, attn = self.gtb_s2(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = rearrange(x, '(b t) s d -> b t s d', t=T, s=S)
        return x, None

class PredFormerEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(PredFormerEncoder, self).__init__()
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

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    def forward(self, x):
        # x [B, T, H, W, C]
        B = x.shape[0]
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        x = self.proj(x) # -> [B, T, D, H//P, W//P]
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B) # -> [B, T, S, D]
        return x

class PatchRecovery(nn.Module):
    def __init__(self, d_model, c_out, patch_size, height, width):
        super(PatchRecovery, self).__init__()
        self.proj = nn.Linear(d_model, patch_size*patch_size*c_out)
        self.patch_size = patch_size
        self.height = height
        self.width = width

    def forward(self, x):
        x = self.proj(x) # -> [B, T, S, P*P*C]
        x = rearrange(x, 'b t (h w) c -> b t h w c', h=self.height//self.patch_size, w=self.width//self.patch_size)
        x = rearrange(x, 'b t h w (p1 p2 c) -> b t (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)
        return x


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
        self.attn_type = configs.attn_type
        self.attn_types = ['Full', 'FacTS', 'FacST', 'BinaryTS', 'BinaryST', 'TripletTST', 'TripletSTS', 'QuadrupletTSST', 'QuadrupletSTTS']
        assert self.attn_type in self.attn_types, f"attn_type must be one of {self.attn_types}"
        assert self.height % self.patch_size == 0, "Height must be divisible by patch size"
        assert self.width % self.patch_size == 0, "Width must be divisible by patch size"

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, configs.enc_in, configs.d_model)

        # self.positional_encoding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model))
        H, W = self.height // self.patch_size, self.width // self.patch_size
        self.positional_encoding = PositionalEmbedding2D(configs.d_model, H, W)

        if self.attn_type == 'Full':
            self.encoder = PredFormerEncoder([
                FullAttentionLayer(
                    attention=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'BinaryTS':
            self.encoder = PredFormerEncoder([
                BinaryTSLayer(
                    attention_t=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'BinaryST':
            self.encoder = PredFormerEncoder([
                BinarySTLayer(
                    attention_s=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'TripletTST':
            self.encoder = PredFormerEncoder([
                TripletTSTLayer(
                    attention_t1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'TripletSTS':
            self.encoder = PredFormerEncoder([
                TripletSTSLayer(
                    attention_s1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'QuadrupletTSST':
            self.encoder = PredFormerEncoder([
                QuadrupletTSSTLayer(
                    attention_t1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        elif self.attn_type == 'QuadrupletSTTS':
            self.encoder = PredFormerEncoder([
                QuadrupletSTTSLayer(
                    attention_s1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t1=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_t2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    attention_s2=AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for _ in range(configs.e_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model))

        self.patch_recovery = PatchRecovery(configs.d_model, configs.c_out, configs.patch_size, configs.height, configs.width)

    def forward_core(self, x_enc):
        # x_enc [B, T, H, W, C]
        
        # patch embedding
        x_enc = self.patch_embed(x_enc) # -> [B, T, S, D]

        # encoding
        x_enc, attns = self.encoder(x_enc, attn_mask=None, tau=None, delta=None)

        # patch recovery
        dec_out = self.patch_recovery(x_enc)

        return dec_out, attns

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