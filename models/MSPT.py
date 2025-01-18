import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_inverted, PositionalEmbedding, PositionalEmbedding2D
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention

import numpy as np
from math import sqrt, ceil
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import seaborn as sns


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
        stitched = torch.einsum("bcld,bc -> bcld", stitched, _nonzero_gates)
    zeros = torch.zeros(gates.size(0),
                        expert_out[-1].size(1),
                        expert_out[-1].size(2),
                        expert_out[-1].size(3),
                        requires_grad=True,
                        device=stitched.device)
    # combine samples that have been processed by the same k experts
    combined = zeros.index_add(0, _batch_index, stitched.float())
    # add eps to all zero values in order to avoid nans when going back to log space
    combined[combined == 0] = np.finfo(float).eps
    # back to log space
    return combined.log()


class MultiScalePeriodicPatchEmbedding(nn.Module):

    def __init__(self,
                 seq_len,
                 num_features,
                 top_k=5,
                 d_model=512,
                 dropout=0.,
                 adaptive=True,
                 use_periodicity=True):
        super(MultiScalePeriodicPatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        # get the patch sizes
        self.patch_sizes = self.get_patch_sizes(seq_len)
        # AFNO1D parameters
        self.start_fc = nn.Linear(num_features, 1)
        self.num_freqs = seq_len // 2

        self.scale = 1 / d_model
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_freqs, self.num_freqs * 4))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs * 4))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_freqs * 4, self.num_freqs))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_freqs))
        # Noise parameters
        self.w_gate = nn.Parameter(
            torch.zeros(self.num_freqs, len(self.patch_sizes)))
        self.w_noise = nn.Parameter(
            torch.zeros(self.num_freqs, len(self.patch_sizes)))
        # Patch Embedding parameters
        self.value_embeddings = nn.ModuleList()
        self.padding_patch_layers = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.value_embeddings.append(
                nn.Linear(patch_size, d_model, bias=False))
            self.padding_patch_layers.append(
                nn.ReplicationPad1d(
                    (0, ceil(seq_len / patch_size) * patch_size - seq_len)))
        self.position_embedding = PositionalEmbedding2D(
            d_model, num_features, 512)
        # self.position_embedding = PositionalEmbedding(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.adaptive = adaptive
        self.use_periodicity = use_periodicity

    def get_patch_sizes(self, seq_len):
        # get the period list, first element is inf if exclude_zero is False
        peroid_list = 1 / torch.fft.rfftfreq(seq_len)[1:]
        patch_sizes = peroid_list.floor().int().unique().detach().cpu().numpy(
        )[::-1]
        # patch_sizes = peroid_list.ceil().int().unique().detach().cpu().numpy()[::-1]
        print(patch_sizes)
        return patch_sizes

    def afno1d_for_peroid_weights(self, x, training, noise_epsilon=1e-2):
        # x [B, L, C]
        B, L, C = x.shape

        x = self.start_fc(x).squeeze(-1)  # [B, L]

        # if self.adaptive:

        # x = rearrange(x, 'B L C -> B C L') # [B, C, L]
        xf = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, L//2+1]
        # xf = torch.fft.rfft(x, dim=-1) # [B, L//2+1]
        xf_ac = xf[:, 1:]  # [B, L//2]

        o1_real = F.relu(xf_ac.real @ self.w1[0] - xf_ac.imag @ self.w1[1] +
                         self.b1[0])
        o1_imag = F.relu(xf_ac.imag @ self.w1[0] + xf_ac.real @ self.w1[1] +
                         self.b1[1])
        o2_real = o1_real @ self.w2[0] - o1_imag @ self.w2[1] + self.b2[0]
        o2_imag = o1_imag @ self.w2[0] + o1_real @ self.w2[1] + self.b2[1]

        xf_ac = torch.stack([o2_real, o2_imag], dim=-1)  # [B, L-1, 2]
        xf_ac = F.softshrink(xf_ac, lambd=0.01)  # [B, L-1, 2]
        xf_ac = torch.view_as_complex(xf_ac)  # [B, L-1]
        xf_ac = torch.abs(xf_ac)  # [B, L-1]

        clean_logits = xf_ac @ self.w_gate
        if training:
            raw_noise_stddev = xf_ac @ self.w_noise
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            logits = noisy_logits  # [B, L-1]
        else:
            logits = clean_logits  # [B, L-1]

        weights = logits  # B, L-1
        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(weights.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/weights.png')

        top_weights, top_indices = torch.topk(weights, self.top_k,
                                              dim=-1)  # [B, top_k]
        top_weights = F.softmax(top_weights, dim=-1)  # [B, top_k]

        zeros = torch.zeros_like(weights)  # [B, Ps]
        gates = zeros.scatter_(-1, top_indices, top_weights)  # [B, Ps]

        # visual gates
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(gates.cpu().detach().numpy(), cmap='viridis')
        # plt.savefig('/root/MSPT/test_visuals/gates.png')

        return gates  # [B, Ps]

    def patch_embedding(self, x, patch_size, index_of_patch):
        B, L, C = x.shape
        # do patching
        x = rearrange(x, 'B L C -> B C L')  # [B, C, L]
        x = self.padding_patch_layers[index_of_patch](x)
        x = x.unfold(-1, patch_size,
                     patch_size)  # [B, C, L//patch_size, patch_size]
        # x = self.value_embeddings[index_of_patch](x) + self.position_embedding(x.permute(0, 2, 1, 3)).unsqueeze(1).repeat(1, 11, 1, 1)
        x = self.value_embeddings[index_of_patch](x) + self.position_embedding(
            x)
        return self.dropout(x)  # [B, C, L, D]

    def forward(self, x):
        gates = self.afno1d_for_peroid_weights(x, self.training)  # [B, Ps]
        xs = dispatch(x, gates)
        _xs = []
        for i, patch_size in enumerate(self.patch_sizes):
            _xs.append(self.patch_embedding(xs[i], patch_size, i))
        return _xs, gates  # Ps*[B, C, L, D], [bs, Ps]


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


class CrossDimensionAttentionLayer(nn.Module):

    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super(CrossDimensionAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, C, L, D = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # attention
        out, attn = self.inner_attention(queries,
                                         keys,
                                         values,
                                         attn_mask,
                                         tau=tau,
                                         delta=delta)

        return self.out_projection(out), attn


class InterPeriodicityAttentionLayer(nn.Module):

    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super(InterPeriodicityAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, C, L, D = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = rearrange(queries, 'B C L D -> B L C D')
        keys = rearrange(keys, 'B C S D -> B S C D')
        values = rearrange(values, 'B C S D -> B S C D')

        # attention
        out, attn = self.inner_attention(queries,
                                         keys,
                                         values,
                                         attn_mask,
                                         tau=tau,
                                         delta=delta)

        out = rearrange(out, 'B L C D -> B C L D')

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):

    def __init__(self,
                 cross_dimension_attention,
                 inter_periodic_attention,
                 d_model,
                 d_ff=None,
                 dropout=0.1,
                 activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_dimension_attention = cross_dimension_attention
        self.inter_periodicity_attention = inter_periodic_attention
        self.cross_dimension_mlp = MLP(d_model, d_ff, dropout, activation)
        self.inter_periodicity_mlp = MLP(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        B, C, L, D = x.shape
        res = x
        x, attn = self.cross_dimension_attention(x,
                                                 x,
                                                 x,
                                                 attn_mask=attn_mask,
                                                 tau=tau,
                                                 delta=delta)
        x = self.norm1(res + self.dropout(x))

        # res = x
        # x = self.cross_dimension_mlp(x)
        # x = self.norm2(res + self.dropout(x))

        res = x
        x, attn = self.inter_periodicity_attention(x,
                                                   x,
                                                   x,
                                                   attn_mask=attn_mask,
                                                   tau=tau,
                                                   delta=delta)
        x = self.norm3(res + self.dropout(x))

        res = x
        x = self.inter_periodicity_mlp(x)
        x = self.norm4(res + self.dropout(x))

        return x, attn


class Encoder(nn.Module):

    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, C, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class LinearPredictionHead(nn.Module):

    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            self.linears.append(nn.Linear(d_model, pred_len))
        # self.cspfa = CrossScalePeriodicFeatureAggregator(patch_sizes, seq_len, d_model)

    def forward(self, xs, gates):
        # Ps*[B, C, L, D]
        for i, patch_size in enumerate(self.patch_sizes):
            xs[i] = self.linears[i](self.dropout(
                xs[i][:, :, -1:, :]))  # [B, C, L, D] -> [B, C, P]
        xs = combine(xs, gates)
        xs = rearrange(xs.squeeze(-2), 'B C P -> B P C')  # [B, P, C]
        return xs  # [bs, P, C]


class LinearPredictionHead2(nn.Module):

    def __init__(self, patch_sizes, seq_len, pred_len, d_model, dropout=0.):
        super(LinearPredictionHead2, self).__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, pred_len)

    def forward(self, xs, gates):
        # Ps*[B, C, L, D]
        _xs = []
        for i, patch_size in enumerate(self.patch_sizes):
            _xs.append(xs[i][:, :, -1:, :])
        _xs = combine(_xs, gates)
        _xs = self.linear(self.dropout(_xs.flatten(-2)))
        _xs = rearrange(_xs, 'B C P -> B P C')
        return _xs


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual

        self.msppe = MultiScalePeriodicPatchEmbedding(self.seq_len,
                                                      configs.enc_in,
                                                      configs.top_k,
                                                      d_model=configs.d_model,
                                                      dropout=configs.dropout)
        self.patch_sizes = self.msppe.patch_sizes

        self.encoders = nn.ModuleList()
        for patch_size in self.patch_sizes:
            self.encoders.append(
                Encoder([
                    EncoderLayer(
                        CrossDimensionAttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        InterPeriodicityAttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation)
                    for l in range(configs.e_layers)
                ],
                        norm_layer=nn.LayerNorm(configs.d_model)))

        self.head = LinearPredictionHead2(self.patch_sizes,
                                          self.seq_len,
                                          self.pred_len,
                                          configs.d_model,
                                          dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-scale periodic patch embedding
        xs_enc, gates_enc = self.msppe(x_enc)  # Ps*[B, C, L, D], [B, Ps]
        # Encoder and Decoder
        enc_outs = []
        for i, x_enc in enumerate(xs_enc):
            enc_out, attns = self.encoders[i](x_enc)  # [B, C, VT, D]
            enc_outs.append(enc_out)

        # Head
        dec_out = self.head(enc_outs, gates_enc)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len, 1))

        return dec_out
