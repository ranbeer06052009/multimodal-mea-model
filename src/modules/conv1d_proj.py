import torch
import torch.nn as nn


class Conv1DProjection(nn.Module):

    """
    Temporal feature projection using 1D convolution.
    Converts modality features to a shared embedding dimension.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        kernel_size=3,
        dropout=0.1
    ):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.norm = nn.BatchNorm1d(d_model)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        x : (B, T, input_dim)
        """

        # convert to (B, input_dim, T)

        x = x.transpose(1, 2)

        x = self.conv(x)

        x = self.norm(x)

        x = self.activation(x)

        x = self.dropout(x)

        # convert back to (B, T, d_model)

        x = x.transpose(1, 2)

        return x