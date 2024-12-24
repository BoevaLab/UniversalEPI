from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Linear,
    MaxPool1d,
    Module,
    ReLU,
    ELU,
    LeakyReLU,
    Sequential,
)
from torch.nn.modules import pooling


class DenseLayer(Module):
    def __init__(self, in_channels=4, out_channels=4, dropout=0.2):
        super(DenseLayer, self).__init__()

        self.layer = Sequential(
            Linear(in_channels, out_channels),
            LeakyReLU(0.2),
            Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class ConvLayer(Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        kernel_size=8,
        batchnorm=False,
        pooling=2,
        dropout=0.2,
    ):
        super(ConvLayer, self).__init__()

        self.use_batchnorm = batchnorm

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )

        self.activation = LeakyReLU(0.2)
 
        self.batchnorm = BatchNorm1d(out_channels)

        self.pooling = MaxPool1d(kernel_size=pooling)

        self.dropout = Dropout(dropout)

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x