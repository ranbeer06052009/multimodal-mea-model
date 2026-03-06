import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################
# Exclusive Node Construction Layer
###############################################################

class ExclusiveNodeConstruction(nn.Module):

    """
    Constructs heterogeneous nodes from exclusive BBFN features
    """

    def __init__(self, dim):

        super().__init__()

        self.proj = nn.Linear(dim, dim)

    def forward(self, features):

        """
        features : list of tensors [(B,T,D), (B,T,D), ...]

        Output:
        nodes : (B, num_nodes, D)
        """

        # temporal pooling
        nodes = [f.mean(dim=1) for f in features]

        nodes = torch.stack(nodes, dim=1)

        nodes = self.proj(nodes)

        return nodes


###############################################################
# Agnostic Node Construction Layer
###############################################################

class AgnosticNodeConstruction(nn.Module):

    """
    Constructs homogeneous graph nodes from agnostic features
    """

    def __init__(self, dim):

        super().__init__()

        self.proj = nn.Linear(dim, dim)

    def forward(self, features):

        nodes = [f.mean(dim=1) for f in features]

        nodes = torch.stack(nodes, dim=1)

        nodes = self.proj(nodes)

        return nodes


###############################################################
# Relational Graph Attention Layer (R-GAT)
###############################################################

class RGATLayer(nn.Module):

    """
    Heterogeneous Graph Fusion
    """

    def __init__(self, dim, heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, nodes):

        """
        nodes : (B, N, D)
        """

        attn_out, _ = self.attn(nodes, nodes, nodes)

        nodes = self.norm1(nodes + attn_out)

        ffn_out = self.ffn(nodes)

        nodes = self.norm2(nodes + ffn_out)

        return nodes


###############################################################
# Graph Attention Layer (Homogeneous)
###############################################################

class GATLayer(nn.Module):

    """
    Homogeneous Graph Fusion
    """

    def __init__(self, dim, heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes):

        """
        nodes : (B, N, D)
        """

        attn_out, _ = self.attn(nodes, nodes, nodes)

        nodes = self.norm(nodes + attn_out)

        return nodes


###############################################################
# Cross Space Fusion
###############################################################

class CrossSpaceFusion(nn.Module):

    """
    Fuse exclusive and agnostic graph embeddings
    """

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim * 2, dim)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.2)

    def forward(self, exclusive_nodes, agnostic_nodes):

        """
        exclusive_nodes : (B,N,D)
        agnostic_nodes  : (B,N,D)
        """

        he = exclusive_nodes.mean(dim=1)

        ha = agnostic_nodes.mean(dim=1)

        h = torch.cat([he, ha], dim=-1)

        h = self.fc(h)

        h = self.activation(h)

        h = self.dropout(h)

        return h


###############################################################
# Prediction Layer
###############################################################

class PredictionLayer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, 128),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(128, 1)

        )

    def forward(self, x):

        return self.net(x)


###############################################################
# Full Graph Fusion Module
###############################################################

class GraphFusion(nn.Module):

    """
    Complete Graph Fusion Module
    """

    def __init__(self, dim):

        super().__init__()

        self.exclusive_node = ExclusiveNodeConstruction(dim)

        self.agnostic_node = AgnosticNodeConstruction(dim)

        self.rgat = RGATLayer(dim)

        self.gat = GATLayer(dim)

        self.cross_fusion = CrossSpaceFusion(dim)

        self.predictor = PredictionLayer(dim)

    def forward(self, exclusive_features, agnostic_features):

        """
        exclusive_features : list of BBFN outputs
        agnostic_features  : list of BBFN outputs
        """

        # Node construction

        nodes_e = self.exclusive_node(exclusive_features)

        nodes_a = self.agnostic_node(agnostic_features)

        # Graph attention

        nodes_e = self.rgat(nodes_e)

        nodes_a = self.gat(nodes_a)

        # Cross space fusion

        fused = self.cross_fusion(nodes_e, nodes_a)

        # Prediction

        output = self.predictor(fused)

        return output