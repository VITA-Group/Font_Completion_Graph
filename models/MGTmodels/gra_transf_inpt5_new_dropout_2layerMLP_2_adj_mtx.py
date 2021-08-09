import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_transformer_layers_new_dropout_2_adj_mtx import *


class GraphTransformerEncoder(nn.Module):

    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=6, n_heads=8,
                 embed_dim=512, feedforward_dim=2048, normalization='batch', dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()

        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        # self.in_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            MultiGraphTransformerLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask1=None, attention_mask2=None):
        # Embed inputs to embed_dim
        # h = self.coord_embed(coord) + self.feat_embed(flag) + self.feat_embed(pos)
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag)), dim=2)
        h = torch.cat((h, self.feat_embed(pos)), dim=2)
        # h = self.in_drop(h)

        # Perform n_layers of Graph Transformer blocks
        for layer in self.transformer_layers:
            h = layer(h, mask1=attention_mask1, mask2=attention_mask2)

        return h


class GraphTransformerClassifier(nn.Module):

    def __init__(self, n_classes, coord_input_dim, feat_input_dim, feat_dict_size,
                 n_layers=6, n_heads=8, embed_dim=512, feedforward_dim=2048,
                 normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):

        super(GraphTransformerClassifier, self).__init__()

        self.encoder = GraphTransformerEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, n_layers,
            n_heads, embed_dim, feedforward_dim, normalization, dropout)

        self.mlp_classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
            nn.ReLU(),
            # nn.Dropout(mlp_classifier_dropout),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )

        # self.mlp_classifier = nn.Sequential(
        #     nn.Dropout(mlp_classifier_dropout),
        #     nn.Linear(embed_dim * 3, feedforward_dim, bias=True),
        #     nn.ReLU(),
        #     # TODO
        #     nn.Dropout(mlp_classifier_dropout),
        #     nn.Linear(feedforward_dim, feedforward_dim, bias=True),
        #     nn.ReLU(),
        #     # nn.Dropout(mlp_classifier_dropout),
        #     nn.Linear(feedforward_dim, n_classes, bias=True)
        # )

        # self.g1 = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.g2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, coordinate, flag_bits, stroke_len, attention_masks, position_encoding):
        """
        Args:
            coordinate: Input coordinates (batch_size, seq_length, coord_input_dim)
            # TODO feat: Input features (batch_size, seq_length, feat_input_dim)
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            stroke_len: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """

        attention_mask1, attention_mask2, padding_mask = attention_masks

        # Embed input sequence
        h = self.encoder(coordinate, flag_bits, position_encoding, attention_mask1, attention_mask2)

        # h = torch.sigmoid(self.g1(h)) * self.g2(h)

        # Mask out padding embeddings to zero
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim=1)
            # g = masked_h.sum(dim=1)/true_seq_length.type_as(h)

        else:
            g = h.sum(dim=1)

        # Compute logits
        logits = self.mlp_classifier(g)

        return logits, g


def make_model(n_classes=345, coord_input_dim=2, feat_input_dim=2, feat_dict_size=104,
               n_layers=6, n_heads=8, embed_dim=512, feedforward_dim=2048,
               normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):
    model = GraphTransformerClassifier(
        n_classes, coord_input_dim, feat_input_dim, feat_dict_size, n_layers,
        n_heads, embed_dim, feedforward_dim, normalization, dropout, mlp_classifier_dropout)

    print(model)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    return model
