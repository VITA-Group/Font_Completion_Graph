import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqEncoderConv(nn.Module):

    def __init__(self, n_classes, coord_input_dim, quant_input_dim, feat_dict_size, load_quantize=False,
                 n_layers=6, n_heads=8, input_embed_dim=512, feed_forward_dim=2048,
                 normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):
        super(SeqEncoderConv, self).__init__()
        self.load_quantize = load_quantize

        # Embedding/Input layers
        if load_quantize:
            self.quant_embed = nn.Embedding(quant_input_dim, coord_input_dim)

        self.coord_embed = nn.Linear(coord_input_dim, input_embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, input_embed_dim)
        # self.in_drop = nn.Dropout(dropout)

        self.last_layer = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv1d(input_embed_dim * 3, feed_forward_dim * 3, 5, stride=3, padding=1),
            # torch.Size([64, 1024, 49])
            nn.BatchNorm1d(feed_forward_dim * 3),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv1d(feed_forward_dim * 3, feed_forward_dim * 2, 7, stride=5),  # torch.Size([64, 1024, 9])
            nn.BatchNorm1d(feed_forward_dim * 2),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv1d(feed_forward_dim * 2, feed_forward_dim, 7, stride=5),  # torch.Size([64, 1024, 1])
            nn.BatchNorm1d(feed_forward_dim),
            nn.ReLU(),
        )

        self.mlp_classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(feed_forward_dim, n_classes, bias=True),
        )

    def forward(self, coordinate, flag_bits, stroke_len, attention_masks, position_encoding):
        # Embed inputs to embed_dim
        # h = self.coord_embed(coord) + self.feat_embed(flag) + self.feat_embed(pos)
        if self.load_quantize:
            coordinate = self.quant_embed(coordinate)

        h = torch.cat((self.coord_embed(coordinate), self.feat_embed(flag_bits)), dim=2)
        h = torch.cat((h, self.feat_embed(position_encoding)), dim=2)
        # h = self.in_drop(h)

        # # # Mask out padding embeddings to zero
        # if padding_mask is not None:
        #     h = h * padding_mask.type_as(h)

        # add one layer to map feature to feedforward_dim
        h = h.permute(0, 2, 1)  # torch.Size([64, 768, 250])
        h = self.last_layer(h)
        h = h.squeeze(-1)

        # Compute logits
        logits = self.mlp_classifier(h)

        return logits, h
