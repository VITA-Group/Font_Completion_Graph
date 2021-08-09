import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import MobileNetV2


class ImgEncoderConv(nn.Module):

    def __init__(self, n_classes, input_embed_dim=512, feed_forward_dim=2048,
                 mlp_classifier_dropout=0.1, conv_embed_dim=8):
        super(ImgEncoderConv, self).__init__()

        # torch.Size([64, 1, 128, 128])
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, conv_embed_dim, stride=2, padding=3),
            # torch.Size([64, 16, 64, 64])
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv2d(16, 64, conv_embed_dim, stride=2, padding=3),
            # torch.Size([64, 64, 32, 32])
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv2d(64, 256, conv_embed_dim, stride=2, padding=3),
            # torch.Size([64, 256, 16, 16])
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv2d(256, 256, conv_embed_dim, stride=2, padding=3),
            # torch.Size([64, 256, 8, 8])
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(mlp_classifier_dropout),
            nn.Conv2d(256, feed_forward_dim, conv_embed_dim, stride=1),
            # torch.Size([64, 1024, 1, 1])
            nn.BatchNorm2d(feed_forward_dim),
            nn.ReLU(),
        )

        self.mlp_classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(feed_forward_dim, n_classes, bias=True),
        )

    def forward(self, sketch):
        # h = h.permute(0, 2, 1)  # torch.Size([64, 768, 250])
        h = self.conv_layer(sketch)
        h = h.squeeze()

        # Compute logits
        logits = self.mlp_classifier(h)

        return logits, h


class MobileNet_modified(MobileNetV2):

    def __init__(self, n_classes=1000,
                 width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None,
                 mlp_classifier_dropout=0.1, embed_dim=512, feedforward_dim=2048):
        super().__init__(n_classes, width_mult, inverted_residual_setting, round_nearest, block)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(self.last_channel, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        y = self.classifier(x)
        return y, x


class Classifier_only(nn.Module):

    def __init__(self, n_classes, last_channel, mlp_classifier_dropout=0.1, feedforward_dim=2048):
        super().__init__()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(mlp_classifier_dropout),
            nn.Linear(self.last_channel, n_classes),
        )

    def forward(self, x):
        y = self.classifier(x)
        return y, x
