# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)



class EMGTDSConv2dBlock(nn.Module):
    """
    A TDS convolution block modified for EMG signal processing with:
    1. Dilated convolutions for better temporal receptive field
    2. Channel-temporal attention mechanism for focusing on relevant EMG patterns
    """

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int,
        dilation_rate: int = 2,
        window_size: int = 32
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.window_size=window_size

        self.conv2d = nn.Conv2d( # adding dialation to increase context length
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, dilation_rate), 
            padding=(0, (kernel_width - 1) * dilation_rate // 2), 
        )

        
        self.channel_attn = nn.Sequential(  # network within network idea to connect features across channels (spatial)
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 1D conv over time
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, self.window_size), padding="same"),
            nn.Sigmoid(), # forces output to be (0,1), making this a sort of priority filter (like the prority gate in GRU).
        )

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)

        x = self.conv2d(x) # conv is now dialted
        x = self.relu(x)
            
        x = x * self.channel_attn(x)

        # Temporal attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # A prority gate over time segements in past.
        x = x * self.temporal_attn(avg_pool) # element wise

        # Reshape back to TNC format
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC

class TDSLSTMEncoder(nn.Module):
    """
    """

    def __init__(
        self,
        num_features: int,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 4
    ) -> None:
        super().__init__()

        self.lstm_layers = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=False, # Input shape: (T, N, num_features)
            bidirectional=True
        )

        # Fully connected block (remains the same)
        self.fc_block = TDSFullyConnectedBlock(lstm_hidden_size * 2)
        self.out_layer = nn.Linear(lstm_hidden_size * 2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm_layers(inputs) # (T, N, lstm_hidden_size * 2)
        x = self.fc_block(x) # Apply FC Transformation
        x = self.out_layer(x) 
        # print('Shape after TDSLSTMEncoder:', x.shape)
        return x


class StackedLSTMGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(StackedLSTMGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)

        # Intermediate Linear + Activation
        self.intermediate_linear = TDSFullyConnectedBlock(hidden_size)

        # GRU Layer(s)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM Forward Pass
        lstm_out, _ = self.lstm(x)

        # Intermediate Layer Pass
        lstm_out = self.intermediate_linear(lstm_out)

        # GRU Forward Pass
        gru_out, _ = self.gru(lstm_out)

        # Output Layer (using the last time step's output)
        out = self.fc(gru_out)  # Take the last time step output

        return out

    

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding1D, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        length = x.shape[1]  # Dynamically determine length
        pe = torch.zeros(length, self.d_model, device=x.device)
        position = torch.arange(0, length, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) *
             -(math.log(10000.0) / self.d_model))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return self.dropout(x + pe)

    

class EMGTransformer(nn.Module):
    def __init__(self, in_features, length, d_model, nhead, num_encoder_layers, num_classes):
        super(EMGTransformer, self).__init__()

        self.d_model = d_model

        # Feature Projection from Features -> D-Model
        self.feature_projection = nn.Linear(in_features, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.mlp = TDSFullyConnectedBlock(d_model)
        
        # Final classification
        self.classifier = nn.Linear(d_model, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        # x shape: [sequence, batch, features]
        x = self.feature_projection(x)

        # Add 2D positional encoding
        self.pos_encoder = PositionalEncoding1D(self.d_model).to(x.device)

        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply Transformer
        x = self.transformer_encoder(x)
        
        # Apply FCC
        x = self.mlp(x)
        
        # Classification
        x = self.classifier(x)
        x = self.softmax(x)
        
        return x
