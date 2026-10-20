import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        return self.linear(x)


class ResidualGCNBlock(nn.Module):

    def __init__(self, channels, dropout=0.2):
        super().__init__()

        self.conv1 = GraphConv(channels, channels)
        self.conv2 = GraphConv(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):

        residual = x

        x = self.conv1(x, adj_norm)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x, adj_norm)
        x = self.norm2(x)

        x = x + residual
        x = F.gelu(x)

        return x


class SSVEPGNNv2NoGraph(nn.Module):
    """
    Frequency-preserving temporal GNN.

    Input:
        [B, 9, T]

    Output:
        [B, 40]

    The temporal representation is deliberately richer than v1.
    """

    def __init__(
        self,
        num_channels=9,
        num_classes=40,
        graph_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        # ---------------------------------------------------------
        # Multi-scale temporal filters
        # ---------------------------------------------------------

        # Short temporal receptive field
        self.temporal_short = nn.Sequential(
            nn.Conv1d(
                1,
                24,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )

        # Medium temporal receptive field
        self.temporal_medium = nn.Sequential(
            nn.Conv1d(
                1,
                24,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )

        # Long temporal receptive field
        self.temporal_long = nn.Sequential(
            nn.Conv1d(
                1,
                24,
                kernel_size=31,
                padding=15,
                bias=False,
            ),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Second temporal stage
        # ---------------------------------------------------------

        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(
                72,
                96,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )

        # Preserve substantially more temporal information than v1
        self.pool = nn.AdaptiveAvgPool1d(32)

        # ---------------------------------------------------------
        # Node projection
        # ---------------------------------------------------------

        self.node_projection = nn.Sequential(
            nn.Linear(
                96 * 32,
                graph_dim,
            ),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Graph network
        # ---------------------------------------------------------

        self.gcn_input = GraphConv(
            graph_dim,
            graph_dim,
        )

        self.gcn_block1 = ResidualGCNBlock(
            graph_dim,
            dropout=dropout,
        )

        self.gcn_block2 = ResidualGCNBlock(
            graph_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Electrode attention
        # ---------------------------------------------------------

        self.node_attention = nn.Sequential(
            nn.Linear(
                graph_dim,
                graph_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                graph_dim // 2,
                1,
            ),
        )

        # ---------------------------------------------------------
        # Classifier
        # ---------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                graph_dim,
                graph_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                graph_dim,
                num_classes,
            ),
        )

    def forward(self, x, adj_norm):

        B, N, T = x.shape

        assert N == self.num_channels

        # Each electrode is processed independently
        x = x.reshape(
            B * N,
            1,
            T,
        )

        # Multi-scale temporal extraction
        short = self.temporal_short(x)
        medium = self.temporal_medium(x)
        long = self.temporal_long(x)

        # Preserve information from all temporal scales
        x = torch.cat(
            [short, medium, long],
            dim=1,
        )

        x = self.temporal_fusion(x)

        x = self.pool(x)

        # Flatten temporal representation
        x = x.flatten(start_dim=1)

        # Convert each electrode into a graph node feature
        x = self.node_projection(x)

        x = x.reshape(
            B,
            N,
            -1,
        )

        # ---------------------------------------------------------
        # GNN
        # ---------------------------------------------------------

        x = self.gcn_input(
            x,
            adj_norm,
        )

        x = F.gelu(x)

        x = self.gcn_block1(
            x,
            adj_norm,
        )

        x = self.gcn_block2(
            x,
            adj_norm,
        )

        # ---------------------------------------------------------
        # Attention pooling
        # ---------------------------------------------------------

        scores = self.node_attention(x)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        x = torch.sum(
            weights * x,
            dim=1,
        )

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------

        return self.classifier(x)


def normalize_adjacency(adj):

    N = adj.shape[0]

    identity = torch.eye(
        N,
        device=adj.device,
        dtype=adj.dtype,
    )

    adj_hat = adj + identity

    degree = adj_hat.sum(dim=1)

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5,
    )

    degree_inv_sqrt[
        torch.isinf(degree_inv_sqrt)
    ] = 0.0

    D = torch.diag(
        degree_inv_sqrt
    )

    return D @ adj_hat @ D


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(
        8,
        9,
        250,
    )

    adj = torch.ones(9, 9) - torch.eye(9)

    adj_norm = normalize_adjacency(adj)

    model = SSVEPGNNv2()

    output = model(
        x,
        adj_norm,
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print("Parameters  :", parameters)
