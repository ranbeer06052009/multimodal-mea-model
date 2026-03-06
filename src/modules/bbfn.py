import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################
# Temporal Transformer
###############################################

class TemporalTransformer(nn.Module):

    def __init__(self, dim, heads=4, layers=1):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim*4,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x):

        # x : (B,T,D)

        return self.encoder(x)



###############################################
# Transformer Block (Attention + FFN)
###############################################

class TransformerBlock(nn.Module):

    def __init__(self, dim, heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):

        attn_out,_ = self.attn(q,k,v)

        x = self.norm1(q + attn_out)

        ffn_out = self.ffn(x)

        x = self.norm2(x + ffn_out)

        return x



###############################################
# Retain Gate
###############################################

class RetainGate(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim, dim)

    def forward(self, x):

        gate = torch.sigmoid(self.fc(x))

        return x * gate



###############################################
# Compound Gate
###############################################

class CompoundGate(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim*2, dim)

    def forward(self, x1, x2):

        g = torch.sigmoid(self.fc(torch.cat([x1,x2],dim=-1)))

        return g * x1 + (1-g) * x2



###############################################
# Modality Specific Feature Separator
###############################################

class FeatureSeparator(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim*2, dim)

    def forward(self, x1, x2):

        joint = torch.cat([x1,x2],dim=-1)

        c_hat = torch.tanh(self.fc(joint))

        return c_hat



###############################################
# BBFN Block
###############################################

class BBFNBlock(nn.Module):

    """
    Bidirectional Bimodal Fusion Block
    """

    def __init__(self, dim, heads=4):

        super().__init__()

        # Cross attention
        self.attn_m1 = TransformerBlock(dim,heads)
        self.attn_m2 = TransformerBlock(dim,heads)

        # Temporal modeling (REPLACES BiGRU)
        self.temporal_m1 = TemporalTransformer(dim)
        self.temporal_m2 = TemporalTransformer(dim)

        # Gates
        self.retain_m1 = RetainGate(dim)
        self.retain_m2 = RetainGate(dim)

        self.compound_m1 = CompoundGate(dim)
        self.compound_m2 = CompoundGate(dim)

        # Feature separator
        self.separator = FeatureSeparator(dim)

    def forward(self, xm1, xm2):

        """
        xm1, xm2 : (B,T,D)
        """

        # Cross-modal attention
        h1 = self.attn_m1(xm1, xm2, xm2)
        h2 = self.attn_m2(xm2, xm1, xm1)

        # Temporal modeling
        t1 = self.temporal_m1(h1)
        t2 = self.temporal_m2(h2)

        # Retain gates
        r1 = self.retain_m1(t1)
        r2 = self.retain_m2(t2)

        # Compound gates
        c1 = self.compound_m1(r1, xm1)
        c2 = self.compound_m2(r2, xm2)

        # Modality-specific separator
        c_hat = self.separator(c1,c2)

        # Updated features
        xm1_i = c1 + c_hat
        xm2_i = c2 + c_hat

        return xm1_i, xm2_i, c_hat