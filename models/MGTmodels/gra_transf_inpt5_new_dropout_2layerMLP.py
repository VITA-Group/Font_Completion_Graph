import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_transformer_layers_new_dropout import *


class GraphTransformerEncoder(nn.Module):

    def __init__(self, coord_input_dim, quant_input_dim, feat_dict_size, n_layers=6, n_heads=8,
                 embed_dim=512, normalization='batch', dropout=0.1, load_quantize=False):
        super(GraphTransformerEncoder, self).__init__()
        self.load_quantize = load_quantize

        # Embedding/Input layers
        if load_quantize:
            self.quant_embed = nn.Embedding(quant_input_dim, coord_input_dim)

        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        # self.in_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(n_heads, embed_dim * 3, normalization, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, coordinate, flag_bits, position_encoding, attention_mask=None):
        # Embed inputs to embed_dim
        # h = self.coord_embed(coord) + self.feat_embed(flag) + self.feat_embed(pos)
        if self.load_quantize:
            coordinate = self.quant_embed(coordinate)

        h = torch.cat((self.coord_embed(coordinate), self.feat_embed(flag_bits)), dim=2)
        h = torch.cat((h, self.feat_embed(position_encoding)), dim=2)
        # h = self.in_drop(h)

        # Perform n_layers of Graph Transformer blocks
        for layer in self.transformer_layers:
            h = layer(h, mask=attention_mask)

        return h


# modified on 2019 10 23.
class GraphTransformerClassifier(nn.Module):

    def __init__(self, n_classes, coord_input_dim, quant_input_dim, feat_dict_size, load_quantize=False,
                 n_layers=6, n_heads=8, input_embed_dim=512, feed_forward_dim=2048,
                 normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):
        super(GraphTransformerClassifier, self).__init__()
        self.load_quantize = load_quantize

        self.encoder = GraphTransformerEncoder(
            coord_input_dim, quant_input_dim, feat_dict_size, n_layers,
            n_heads, input_embed_dim, normalization, dropout, load_quantize)

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
            position_encoding: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """

        attention_mask, padding_mask = attention_masks

        # Embed input sequence
        h = self.encoder(coordinate, flag_bits, position_encoding, attention_mask)
        # h = torch.sigmoid(self.g1(h)) * self.g2(h)

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


def make_model(n_classes=345, coord_input_dim=2, quant_input_dim=1000, feat_dict_size=104, load_quantize=False,
               n_layers=6, n_heads=8, input_embed_dim=512, feed_forward_dim=2048,
               normalization='batch', dropout=0.1, mlp_classifier_dropout=0.1):
    model = GraphTransformerClassifier(
        n_classes, coord_input_dim, quant_input_dim, feat_dict_size, load_quantize,
        n_layers, n_heads, input_embed_dim, feed_forward_dim, normalization, dropout, mlp_classifier_dropout)

    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters: ', nb_param)

    return model
