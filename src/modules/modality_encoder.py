import torch
import torch.nn as nn

from .conv1d_projection import Conv1DProjection
from .positional_encoding import PositionalEncoding
from .bilstm_encoder import BiLSTMEncoder


class ModalityEncoder(nn.Module):

    """
    Complete modality encoder:
    Conv1D Projection + Positional Encoding + BiLSTM
    """

    def __init__(
        self,
        input_dim,
        d_model=128
    ):

        super().__init__()

        self.conv_projection = Conv1DProjection(
            input_dim=input_dim,
            d_model=d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model
        )

        self.temporal_encoder = BiLSTMEncoder(
            d_model=d_model
        )

    def forward(self, x):

        """
        x : (B, T, input_dim)
        """

        x = self.conv_projection(x)

        x = self.positional_encoding(x)

        x = self.temporal_encoder(x)

        return x