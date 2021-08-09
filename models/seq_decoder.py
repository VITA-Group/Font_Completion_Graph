import torch
import torch.nn as nn

from models.CategoricalConditionalBase import CatCondBatchNorm1d
from models.CategoricalConditionalBase import CatCondConv1d
from models.MGTmodels.graph_transformer_layers_new_dropout import GraphTransformerLayer


class SeqDecoder(nn.Module):
    def __init__(self, feed_forward_dim, nPoints, coord_input_dim,
                 quant_input_dim, flag_input_dim, char_embed_dim,
                 char_embedded, conditional_conv, conditional_BN,
                 mgt_decoder, quant_decoder, flag_decoder, nChars,
                 conv_embed_dim=32, centers=None):
        super(SeqDecoder, self).__init__()

        self.coord_input_dim = coord_input_dim
        self.quant_input_dim = quant_input_dim
        self.flag_input_dim = flag_input_dim

        self.char_embedded = char_embedded
        self.conditional_conv = conditional_conv
        self.conditional_BN = conditional_BN

        self.mgt_decoder = mgt_decoder
        self.quant_decoder = quant_decoder
        self.flag_decoder = flag_decoder

        if char_embedded or conditional_conv or conditional_BN:
            assert nChars

        if char_embedded:
            self.feat_embed = nn.Embedding(nChars, char_embed_dim)

        if conditional_conv:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d((feed_forward_dim + char_embed_dim) if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([64, 768, 4])
                CatCondBatchNorm1d(feed_forward_dim, nChars),
                nn.ReLU(True),
                CatCondConv1d(nChars, feed_forward_dim, 512, 3, stride=1, padding=1),
                # torch.Size([64, 512, 4])
                CatCondBatchNorm1d(512, nChars),
                nn.ReLU(True),
                CatCondConv1d(nChars, 512, 256, 3, stride=1, padding=1),
                # torch.Size([64, 256, 4])
                CatCondBatchNorm1d(256, nChars),
                nn.ReLU(True),
                CatCondConv1d(nChars, 256, nPoints, 3, stride=1, padding=1),
                # torch.Size([64, nPoints, 4])
                CatCondBatchNorm1d(nPoints, nChars),
                # nn.ReLU(True),
                # CatCondConv1d(nChars, nPoints, nPoints, 3, stride=1, padding=1,, nChars),
                # # torch.Size([64, nPoints, 4])
                # # nn.Tanh()
            )
        elif conditional_BN:
            self.decoder = nn.ModuleList([
                nn.ConvTranspose1d((feed_forward_dim + char_embed_dim) if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([64, 768, 4])
                CatCondBatchNorm1d(feed_forward_dim, nChars),
                nn.ReLU(True),
                nn.Conv1d(feed_forward_dim, 512, 3, stride=1, padding=1),
                # torch.Size([64, 512, 4])
                CatCondBatchNorm1d(512, nChars),
                nn.ReLU(True),
                nn.Conv1d(512, 256, 3, stride=1, padding=1),
                # torch.Size([64, 256, 4])
                CatCondBatchNorm1d(256, nChars),
                nn.ReLU(True),
                nn.Conv1d(256, nPoints, 3, stride=1, padding=1),
                # torch.Size([64, nPoints, 4])
                CatCondBatchNorm1d(nPoints, nChars),
                # nn.ReLU(True),
                # nn.Conv1d(nPoints, nPoints, 3, stride=1, padding=1),
                # # torch.Size([64, nPoints, 4])
                # # nn.Tanh()
            ])
        else:
            # input shape torch.Size([64, 1280, 1])
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d((feed_forward_dim + char_embed_dim) if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([64, 1280, 4])
                nn.BatchNorm1d(feed_forward_dim),
                nn.ReLU(True),
                nn.Conv1d(feed_forward_dim, 512, 3, stride=1, padding=1),
                # torch.Size([64, 512, 4])
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Conv1d(512, 256, 3, stride=1, padding=1),
                # torch.Size([64, 256, 4])
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Conv1d(256, nPoints, 3, stride=1, padding=1),
                # torch.Size([64, nPoints, 4])
                nn.BatchNorm1d(nPoints),
                # nn.ReLU(True),
                # nn.Conv1d(nPoints, nPoints, 3, stride=1, padding=1),
                # # torch.Size([64, nPoints, 4])
                # # nn.Tanh()
            )

        if mgt_decoder:
            # Transformer blocks
            self.transformer_layers = nn.ModuleList([
                GraphTransformerLayer(conv_embed_dim, conv_embed_dim, normalization='batch', dropout=0.25)
                for _ in range(4)  # n_layers
            ])

        self.embed2seq = nn.Conv1d(conv_embed_dim, coord_input_dim, 3, stride=1, padding=1)

        if flag_decoder:
            self.embed2flag = nn.Conv1d(conv_embed_dim, flag_input_dim, 3, stride=1, padding=1)

        if quant_decoder:
            self.embed2quant = nn.Linear(conv_embed_dim, quant_input_dim, bias=False)

    def forward(self, g, char, attention_masks=None):
        assert g.shape[0] == char.shape[0]

        if self.char_embedded:
            g = torch.cat((g, self.feat_embed(char)), dim=-1)

        g = g.unsqueeze(-1)  # torch.Size([64, 1, 768])

        if self.conditional_conv or self.conditional_BN:
            for n, _ in enumerate(self.decoder):
                if (isinstance(self.decoder[n], CatCondBatchNorm1d) or
                        isinstance(self.decoder[n], CatCondConv1d)):
                    g = self.decoder[n](g, char)
                else:
                    g = self.decoder[n](g)
        else:
            g = self.decoder(g)  # torch.Size([64, nPoints, 4])

        if self.mgt_decoder:
            attention_mask, padding_mask = attention_masks

            # Perform n_layers of Graph Transformer blocks
            for layer in self.transformer_layers:
                g = layer(g, mask=attention_mask)

        g = g.permute(0, 2, 1)
        coor_pred = self.embed2seq(g).permute(0, 2, 1)
        flag_pred = self.embed2flag(g).permute(0, 2, 1) if self.flag_decoder else None

        g = g.permute(0, 2, 1)
        quant_pred = self.embed2quant(g) if self.quant_decoder else None

        return coor_pred, flag_pred, quant_pred
