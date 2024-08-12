import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Configurable U-Net architecture model.

    Paper ref: https://arxiv.org/abs/1505.04597
    Implementation ref: https://www.kaggle.com/code/heiswicked/pytorch-lightning-unet-segmentation-tumour

    Parameters
    ----------
    input_channels : int
        Size of the channel dimension of the input data.
        Can be 3 for rgb images or 1 for data shaped like a matrix.
    output_channels : int
        Size of the channel dimension of the output data.
        This is analog to number of classes predicted by the last convolutional layer.
    encoder_first_block_channels : int
        Size of the channel dimension in the first processing block.
        The rest of the channels are inferred from this initial value.
    model_depth : int
        Amount of blocks the encoder and the decoder will be composed of.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encoder_first_block_channels: int,
        model_depth: int,
    ):
        super().__init__()
        decoder_first_block_channels = encoder_first_block_channels * (
            2 ** (model_depth - 1)
        )

        self.encoder = Encoder(
            input_channels, encoder_first_block_channels, model_depth
        )
        self.decoder = Decoder(
            output_channels, decoder_first_block_channels, model_depth
        )

    def forward(self, features):
        features, skip_connections = self.encoder(features)
        features = self.decoder(features, skip_connections)

        # x = x.squeeze(1)
        return features

    def get_hidden_space(self, features):
        features, _ = self.encoder(features)

        return features


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 3,
        first_block_channels: int = 64,
        layer_depth: int = 5,
    ):
        super().__init__()

        if layer_depth < 2:
            raise ValueError(
                f"layer_depth is smaller than 2 ({layer_depth}), which is the minimum possible."
            )

        decoder_blocks = []
        n_channels = first_block_channels
        for _ in range(1, layer_depth):
            decoder_blocks.append(
                Block(
                    n_channels,
                    n_channels // 2,
                    n_channels // 2,
                    upsample=True,
                    downsample=False,
                )
            )
            n_channels //= 2

        self.blocks = nn.ModuleList(decoder_blocks)

        self.head = nn.ConvTranspose2d(
            n_channels, output_channels, kernel_size=1, stride=1
        )

    def forward(self, features: torch.Tensor, skip_connections: list[torch.Tensor]):
        # ToDo: match skip_connection shape to the concatenated features
        skip_connections.reverse()

        for block, skip_connection in zip(self.blocks, skip_connections):
            features, _ = block(features, skip_connection)

        features = self.head(features)

        return features


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        first_block_channels: int = 64,
        layer_depth: int = 5,
    ):
        super().__init__()

        if layer_depth < 1:
            raise ValueError(
                f"layer_depth is smaller than 2 ({layer_depth}), which is the minimum possible."
            )

        encoder_blocks = []
        encoder_blocks.append(
            Block(
                input_channels,
                first_block_channels,
                first_block_channels,
                upsample=False,
                downsample=True,
            )
        )

        n_channels = first_block_channels
        for _ in range(1, layer_depth - 1):
            encoder_blocks.append(
                Block(
                    n_channels,
                    n_channels * 2,
                    n_channels * 2,
                    upsample=False,
                    downsample=True,
                )
            )
            n_channels *= 2

        encoder_blocks.append(
            Block(
                n_channels,
                n_channels * 2,
                n_channels * 2,
                upsample=False,
                downsample=False,
                use_batch_norm=False,
            )
        )

        self.blocks = nn.ModuleList(encoder_blocks)

    def forward(self, features: torch.Tensor):
        skip_connections = []

        for block in self.blocks[:-1]:
            skip_connection, features = block(features)

            skip_connections.append(skip_connection)

        features, _ = self.blocks[-1](features)

        return features, skip_connections


class Block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        upsample: bool,
        downsample: bool,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self._upsample = upsample
        self._downsample = downsample
        self._use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.conv2 = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        self.relu = nn.ReLU()

        if self._upsample:
            self.transpose = nn.ConvTranspose2d(
                input_channels, hidden_channels, kernel_size=2, stride=2
            )
        if self._downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm2d(output_channels)

    def forward(
        self, features: torch.Tensor, skip_connection: torch.Tensor | None = None
    ):
        if self._upsample:
            features = self.transpose(features)

        if skip_connection is not None:
            features = torch.cat([features, skip_connection], dim=1)

        features = self.conv1(features)
        features = self.relu(features)

        features = self.conv2(features)
        if self._use_batch_norm:
            features = self.bn(features)
        features = self.relu(features)

        if self._downsample:
            downsampled_features = self.pool(features)
        else:
            downsampled_features = None

        return features, downsampled_features
