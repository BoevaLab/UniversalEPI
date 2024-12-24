import copy
import math
import random

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stg import FeatureSelector
from .transformer_utils import TransformerEncoder, TransformerEncoderLayer


class PositionalEncodingRelative(nn.Module):
    """
    Genomic distance aware positional encoding
    """

    def __init__(self, d_model, batch_size, dropout=0.1, max_len=12001, resolution=500):
        super(PositionalEncodingRelative, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.resolution = resolution
        self.max_len = max_len
        self.batch_size = batch_size

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(20000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos, embed_dim, peu_flg):

        pos = pos[:, :, -1]
        pos = pos - pos[:, 200:201]
        pos = torch.div(pos, self.resolution)
        pos = torch.clip(pos, min=-self.max_len // 2, max=self.max_len // 2)
        pos = pos + self.max_len // 2

        pos = pos.type(torch.IntTensor)
        idx_bs = torch.arange(x.size(0)).unsqueeze(1)

        if peu_flg:
            x = x + self.pe[idx_bs, pos, :] * math.sqrt(embed_dim)
        else:
            x = x + self.pe[idx_bs, pos, :]

        return self.dropout(x)


class InteractionDecoder(nn.Module):
    def __init__(self, channels_in, channels_out, var_flg=False) -> None:
        super().__init__()
        if var_flg:
            channels_out = 2

        self.base_module = nn.Sequential(
            nn.Linear(2 * channels_in, channels_in),
            nn.ReLU(),
            nn.Linear(channels_in, channels_out),
        )

        self.flip_augmentation = False
        interactions = []
        for i in range(100):
            interactions.extend((200 - i, 200 + i, 200 - i - 1, 200 + i))
        interactions.extend((100, 300))
        # can be easily reshaped to [(200-i, 200+i), (200-i-1, 200+i), ...]
        self.interactions = interactions[2:]
        self.interactions_flipped = copy.deepcopy(self.interactions)
        self.interactions_flipped.reverse()

    def forward(self, x, meta=None):
        if self.flip_augmentation and self.training and 0.5 > random.random():
            x = x[:, self.interactions_flipped, :]
            x = einops.rearrange(x, "batch (interactions pair) channels -> batch interactions (pair channels)", pair=2)
            x = x.flip(-2)  # flip interactions axis
            return self.base_module(x).squeeze()

        x = x[:, self.interactions, :]
        x = einops.rearrange(x, "batch (interactions pair) channels -> batch interactions (pair channels)", pair=2)

        return self.base_module(x).squeeze()


class Transformer_Encoder(nn.Module):
    """
    Flexible Encoder Transformer Class
    """

    def __init__(
        self,
        input_dim,
        batch_size,
        input_feat_dims,
        relative_positions=False,
        meta_flg=False,
        stg_flg=False,
        peu_flg=True,
        map_flg=True,
        atac_flg=False,
        var_flg=False,
        max_len=12001,
        pe_res=500,
        seq_len=401,
        binning=1000,
        num_heads=4,
        num_layers=4,
        embed_dim=32,
        hidden_dim=32,
        dropout=0.6,
        sigma=0.5,
        get_attn=False,
        device=None,
    ):
        super().__init__()

        # Parameters
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.input_feat_dims = input_feat_dims

        self.relative_positions = relative_positions
        self.stg_flg = stg_flg
        self.peu_flg = peu_flg
        self.map_flg = map_flg
        self.atac_flg = atac_flg
        self.var_flg = var_flg

        self.seq_len = seq_len
        self.binning = binning

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_len = (seq_len // 2) + 1

        self.embedding_feat0 = nn.Linear(self.input_feat_dims[0], self.input_dim)
        self.embedding_feat1 = nn.Linear(self.input_feat_dims[1], self.input_dim)
        self.embedding_feat2 = nn.Linear(self.input_feat_dims[2], self.input_dim)
        self.embedding_feat3 = nn.Linear(self.input_feat_dims[3], self.input_dim)
        self.embedding_feat4 = nn.Linear(self.input_feat_dims[4], self.input_dim)

        if self.map_flg:
            self.embedding_map = nn.Linear(self.binning, self.embed_dim)

        if self.atac_flg:
            self.embedding_atac = nn.Linear(self.binning, self.embed_dim)

        self.embedding_x = nn.Linear(self.input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncodingRelative(
            d_model=self.hidden_dim, batch_size=batch_size, max_len=max_len, resolution=pe_res
        )

        self.dropout = nn.Dropout(p=dropout)

        if self.stg_flg:
            self.FeatureSelector = FeatureSelector(5, sigma, device)
            self.reg = self.FeatureSelector.regularizer
            self.mu = self.FeatureSelector.mu
            self.sigma = self.FeatureSelector.sigma

        encoder_layers = TransformerEncoderLayer(
            self.hidden_dim,
            self.num_heads,
            self.hidden_dim,
            dropout,
            relative_positions=relative_positions,
            meta_flg=meta_flg,
            batch_first=True,
            norm_first=False,
            get_attn=get_attn,
            seq_len=seq_len,
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)
        self.decoder = InteractionDecoder(self.hidden_dim, 1, self.var_flg)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.embedding_x.weight, mode="fan_in")

    def forward(self, x, map, meta=None, atac=None):

        encoded_x, wts, reg = self.encode(x, map, meta, atac)
        output = self.decoder(encoded_x)

        return output, wts, reg

    def encode(self, x, map, meta=None, atac=None):

        if self.stg_flg:
            x[0] = torch.reshape(x[0], [-1, self.seq_len, self.input_feat_dims[0]])
            x[0] = self.embedding_feat0(x[0]).unsqueeze(-1)

            x[1] = torch.reshape(x[1], [-1, self.seq_len, self.input_feat_dims[1]])
            x[1] = self.embedding_feat1(x[1]).unsqueeze(-1)

            x[2] = torch.reshape(x[2], [-1, self.seq_len, self.input_feat_dims[2]])
            x[2] = self.embedding_feat2(x[2]).unsqueeze(-1)

            x[3] = torch.reshape(x[3], [-1, self.seq_len, self.input_feat_dims[3]])
            x[3] = self.embedding_feat3(x[3]).unsqueeze(-1)

            x[4] = torch.reshape(x[4], [-1, self.seq_len, self.input_feat_dims[4]])
            x[4] = self.embedding_feat4(x[4]).unsqueeze(-1)

            x = torch.concatenate(x[:5], axis=-1)
        else:
            x = torch.reshape(x[3], [-1, self.seq_len, self.input_dim])

        if self.stg_flg:
            x = self.FeatureSelector(x)
            x = torch.sum(x, axis=-1)
            reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        else:
            reg = 0.0

        x = self.embedding_x(x)

        if self.map_flg:
            map = self.embedding_map(map)
            x = torch.concatenate((x, map), axis=-1)

        if self.atac_flg:
            atac = self.embedding_atac(atac)
            x = torch.concatenate((x, atac), axis=-1)

        if not self.peu_flg:
            x = x * math.sqrt(self.hidden_dim)

        if not self.relative_positions:
            x = self.pos_encoder(x, meta, self.hidden_dim, self.peu_flg)

        x, wts = self.transformer_encoder(x, meta=meta)

        return x, wts, reg
