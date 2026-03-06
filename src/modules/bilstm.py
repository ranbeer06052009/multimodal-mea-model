import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):

    """
    Bidirectional LSTM encoder for temporal sequence modeling.
    """

    def __init__(
        self,
        d_model,
        num_layers=1,
        dropout=0.1
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        x : (B, T, d_model)
        """

        output, _ = self.lstm(x)

        output = self.layer_norm(output)

        output = self.dropout(output)

        return output