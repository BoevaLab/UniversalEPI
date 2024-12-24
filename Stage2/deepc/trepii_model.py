import torch
import torch.nn as nn
import torch.nn.functional as F
from deepc.components import ConvLayer, DenseLayer
from torch.nn import Linear


class DeepC(nn.Module):
    def __init__(self, ksizes, channels, poolings):

        super().__init__()
        self.convolution = nn.ModuleList()

        for i in range(len(ksizes)):
            self.convolution.append(
                ConvLayer(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=ksizes[i],
                    batchnorm=False,
                    pooling=poolings[i],
                    dropout=0.2,
                )
            )

        self.classifier = nn.ModuleList()

        self.classifier.append(
            DenseLayer(
                in_channels=channels[-3],
                out_channels=channels[-2],
                dropout=0.2,
            )
        )

        self.output_layer = Linear(channels[-2], channels[-1])

    def forward(self, x, feat=False):
        # Apply convolution
        feats = []
        for i, layer in enumerate(self.convolution):
            x = layer(x)
            feat_layer = torch.flatten(x, 1)
            feats.append(feat_layer)

        # Apply classifier
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)

        x = self.output_layer(x)

        if feat:
            return feats
        else:
            return x
