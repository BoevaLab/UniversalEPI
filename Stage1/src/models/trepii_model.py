import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ConvLayer, DenseLayer
from torch.nn import Linear


class DeepC(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.ModuleList()
        self.convolution.append(
            ConvLayer(
                in_channels=5,
                out_channels=30,
                kernel_size=11,  # 11
                batchnorm=False,
                pooling=4,
                dropout=0.2,
            )
        )

        self.convolution.append(
            ConvLayer(
                in_channels=30,
                out_channels=60,
                kernel_size=11,
                batchnorm=False,
                pooling=5,
                dropout=0.2,
            )
        )

        self.convolution.append(
            ConvLayer(
                in_channels=60,
                out_channels=60,
                kernel_size=11,
                batchnorm=False,
                pooling=5,
                dropout=0.2,
            )
        )

        self.convolution.append(
            ConvLayer(
                in_channels=60,
                out_channels=90,
                kernel_size=5,  # 5
                batchnorm=False,
                pooling=4,
                dropout=0.2,
            )
        )

        self.convolution.append(
            ConvLayer(
                in_channels=90,
                out_channels=90,
                kernel_size=5,
                batchnorm=False,
                pooling=2,
                dropout=0.2,
            )
        )

        self.classifier = nn.ModuleList()

        self.classifier.append(
            DenseLayer(
                in_channels=90,
                out_channels=90,
                dropout=0.2,
            )
        )

        self.output_layer = Linear(90, 4)

    def forward(self, x, feat=False):
        # Apply convolution
        # for layer in self.convolution:
        feats = []
        for i, layer in enumerate(self.convolution):
            x = layer(x)
            # print(x.shape)
            # if i > 0:
            feat_layer = torch.flatten(x, 1)
            feats.append(feat_layer)
            # if i == 2:
            #     feat_layer = x
            # elif i == 3:
            #     feat_layer_2 = x

        # Apply classifier
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)

        x = self.output_layer(x)

        if feat:
            return feats
        else:
            return x

