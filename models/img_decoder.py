import torch
import torch.nn as nn

from models.CategoricalConditionalBase import CatCondBatchNorm2d
from models.CategoricalConditionalBase import CatCondConvTrans2d


class ImgDecoder(nn.Module):
    def __init__(self, feed_forward_dim, char_embed_dim,
                 char_embedded, conditional_conv, conditional_BN,
                 nChars, conv_embed_dim=8):
        super(ImgDecoder, self).__init__()

        self.char_embedded = char_embedded
        self.conditional_conv = conditional_conv
        self.conditional_BN = conditional_BN

        if char_embedded or conditional_conv or conditional_BN:
            assert nChars

        if char_embedded:
            self.feat_embed = nn.Embedding(nChars, char_embed_dim)

        if conditional_conv:
            self.decoder = nn.ModuleList([
                nn.ConvTranspose2d(feed_forward_dim + char_embed_dim if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([64, 1280, 8, 8])
                CatCondBatchNorm2d(feed_forward_dim, nChars),
                nn.ReLU(True),
                CatCondConvTrans2d(nChars, feed_forward_dim, 256, conv_embed_dim, stride=2, padding=3),
                # torch.Size([64, 256, 16, 16])
                CatCondBatchNorm2d(256, nChars),
                nn.ReLU(True),
                CatCondConvTrans2d(nChars, 256, 64, conv_embed_dim, stride=2, padding=3),
                # torch.Size([64, 64, 32, 32])
                CatCondBatchNorm2d(64, nChars),
                nn.ReLU(True),
                CatCondConvTrans2d(nChars, 64, 16, conv_embed_dim, stride=2, padding=3),
                # torch.Size([64, 16, 64, 64])
                CatCondBatchNorm2d(16, nChars),
                nn.ReLU(True),
                CatCondConvTrans2d(nChars, 16, 1, conv_embed_dim, stride=2, padding=3),
                # torch.Size([64, 1, 128, 128])
                nn.Sigmoid()
            ])
        elif conditional_BN:
            self.decoder = nn.ModuleList([
                nn.ConvTranspose2d(feed_forward_dim + char_embed_dim if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([_, 1280, 8, 8])
                CatCondBatchNorm2d(feed_forward_dim, nChars),
                nn.ReLU(True),
                nn.ConvTranspose2d(feed_forward_dim, 256, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 256, 16, 16])
                CatCondBatchNorm2d(256, nChars),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 64, 32, 32])
                CatCondBatchNorm2d(64, nChars),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 16, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 16, 64, 64])
                CatCondBatchNorm2d(16, nChars),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 1, 128, 128])
                nn.Sigmoid()
            ])
        else:
            # input shape torch.Size([64, 768, 1, 1])
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(feed_forward_dim + char_embed_dim if char_embedded else feed_forward_dim,
                                   feed_forward_dim, conv_embed_dim, stride=1),
                # torch.Size([_, 1280, 8, 8])
                nn.BatchNorm2d(feed_forward_dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(feed_forward_dim, 256, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 256, 16, 16])
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 64, 32, 32])
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 16, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 16, 64, 64])
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, conv_embed_dim, stride=2, padding=3),
                # torch.Size([_, 1, 128, 128])
                nn.Sigmoid()
            )

    def forward(self, g, char):
        assert g.shape[0] == char.shape[0]

        if self.char_embedded:
            g = torch.cat((g, self.feat_embed(char)), dim=-1)

        g = g.unsqueeze(-1).unsqueeze(-1)  # torch.Size([64, 768, 1, 1])

        if self.conditional_conv or self.conditional_BN:
            for n, _ in enumerate(self.decoder):
                if (isinstance(self.decoder[n], CatCondBatchNorm2d) or
                        isinstance(self.decoder[n], CatCondConvTrans2d)):
                    g = self.decoder[n](g, char)
                else:
                    g = self.decoder[n](g)
        else:
            g = self.decoder(g)  # torch.Size([64, 1, 128, 128])

        return g
