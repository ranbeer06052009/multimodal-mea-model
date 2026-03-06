import torch
import torch.nn as nn


###############################################################
# Modality Reinforcement Unit (MRU)
###############################################################

class MRU(nn.Module):

    """
    Cross-modal reinforcement block
    """

    def __init__(self, dim, heads=4):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, k, v):

        q_norm = self.norm1(q)

        attn_out, _ = self.cross_attn(q_norm, k, v)

        x = q + attn_out

        x = self.norm2(x)

        ffn_out = self.ffn(x)

        x = x + ffn_out

        x = self.norm3(x)

        return x


###############################################################
# HCA for Language Branch
###############################################################

class HCA_L(nn.Module):

    """
    Hierarchical cross-modal attention for language
    """

    def __init__(self, dim):

        super().__init__()

        self.mru_lv = MRU(dim)
        self.mru_la = MRU(dim)

    def forward(self, zl, zv, za):

        zl_v = self.mru_lv(zl, zv, zv)

        zl_va = self.mru_la(zl_v, za, za)

        return zl_va


###############################################################
# HCA for Vision Branch
###############################################################

class HCA_V(nn.Module):

    """
    Hierarchical cross-modal attention for vision
    """

    def __init__(self, dim):

        super().__init__()

        self.mru_vl = MRU(dim)
        self.mru_va = MRU(dim)

    def forward(self, zl, zv, za):

        zv_l = self.mru_vl(zv, zl, zl)

        zv_la = self.mru_va(zv_l, za, za)

        return zv_la


###############################################################
# HCA for Audio Branch
###############################################################

class HCA_A(nn.Module):

    """
    Hierarchical cross-modal attention for audio
    """

    def __init__(self, dim):

        super().__init__()

        self.mru_al = MRU(dim)
        self.mru_av = MRU(dim)

    def forward(self, zl, zv, za):

        za_l = self.mru_al(za, zl, zl)

        za_lv = self.mru_av(za_l, zv, zv)

        return za_lv