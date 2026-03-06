import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################
# Predictive Attention Map
#############################################

class PredictiveAttention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x : (B,T,D)

        x_t = x.transpose(1,2)

        attn = self.conv(x_t)

        attn = self.sigmoid(attn)

        attn = attn.transpose(1,2)

        return attn



#############################################
# PSA Layer
#############################################

class PSALayer(nn.Module):

    def __init__(self, dim, heads=8):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.pred_map = PredictiveAttention(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):

        # Self attention

        x_norm = self.norm1(x)

        attn_out,_ = self.attn(x_norm,x_norm,x_norm)

        # Predictive attention weighting

        p_map = self.pred_map(attn_out)

        attn_out = attn_out * p_map

        x = x + attn_out

        x = self.norm2(x)

        # Feed Forward

        ffn_out = self.ffn(x)

        x = x + ffn_out

        x = self.norm3(x)

        return x



#############################################
# PSA Module (2 layers stacked)
#############################################

class PSA(nn.Module):

    def __init__(self, dim, heads=8, layers=2):

        super().__init__()

        self.layers = nn.ModuleList(
            [PSALayer(dim,heads) for _ in range(layers)]
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x