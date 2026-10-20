import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Simple spectral-style GCN layer.

    Input:
        x: [B, N, F_in]

    Output:
        [B, N, F_out]
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        # Spatial message passing
        x = torch.matmul(adj_norm, x)

        # Feature transformation
        x = self.linear(x)

        return x


class ResidualGCNBlock(nn.Module):
    """
    Two-layer residual graph convolution block.
    """

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


class SSVEPGNN(nn.Module):
    """
    GNN classifier for zero-shot SSVEP classification.

    Input:
        EEG tensor: [B, 9, T]

    Output:
        logits: [B, 40]

    Nodes:
        9 selected electrodes

    Classes:
        40 SSVEP targets
    """

    def __init__(
        self,
        num_channels=9,
        num_classes=40,
        temporal_dim=64,
        graph_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        # ---------------------------------------------------------
        # Temporal feature extraction
        # ---------------------------------------------------------
        #
        # Each electrode is processed independently.
        #
        # [B, 9, T]
        #       ?
        # [B*9, 1, T]
        #       ?
        # temporal convolution
        #       ?
        # [B, 9, temporal_dim, T']
        #

        self.temporal = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=15,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(
                64,
                temporal_dim,
                kernel_size=31,
                padding=15,
                bias=False,
            ),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(16),
        )

        # Project temporal representation into graph feature space
        self.node_projection = nn.Linear(
            temporal_dim * 16,
            graph_dim,
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
            nn.Linear(graph_dim, graph_dim // 2),
            nn.GELU(),
            nn.Linear(graph_dim // 2, 1),
        )

        # ---------------------------------------------------------
        # Classifier
        # ---------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, num_classes),
        )

    def forward(self, x, adj_norm):
        """
        Parameters
        ----------
        x:
            EEG tensor [B, 9, T]

        adj_norm:
            normalized graph adjacency [9, 9]

        Returns
        -------
        logits:
            [B, 40]
        """

        B, N, T = x.shape

        assert N == self.num_channels, (
            f"Expected {self.num_channels} channels, "
            f"got {N}"
        )

        # ---------------------------------------------------------
        # Temporal processing per electrode
        # ---------------------------------------------------------

        x = x.reshape(B * N, 1, T)

        x = self.temporal(x)

        # [B*N, temporal_dim, 16]
        x = x.flatten(start_dim=1)

        # [B*N, graph_dim]
        x = self.node_projection(x)

        # [B, N, graph_dim]
        x = x.reshape(B, N, -1)

        # ---------------------------------------------------------
        # Graph processing
        # ---------------------------------------------------------

        x = self.gcn_input(x, adj_norm)
        x = F.gelu(x)

        x = self.gcn_block1(x, adj_norm)
        x = self.gcn_block2(x, adj_norm)

        # ---------------------------------------------------------
        # Attention pooling across electrodes
        # ---------------------------------------------------------

        attention_scores = self.node_attention(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        x = torch.sum(
            attention_weights * x,
            dim=1,
        )

        # ---------------------------------------------------------
        # 40-class prediction
        # ---------------------------------------------------------

        logits = self.classifier(x)

        return logits


def normalize_adjacency(adj):
    """
    Symmetric GCN normalization.

    A_hat = A + I

    D^(-1/2) A_hat D^(-1/2)
    """

    device = adj.device
    dtype = adj.dtype

    N = adj.shape[0]

    identity = torch.eye(
        N,
        device=device,
        dtype=dtype,
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

    D_inv_sqrt = torch.diag(
        degree_inv_sqrt
    )

    adj_norm = (
        D_inv_sqrt
        @ adj_hat
        @ D_inv_sqrt
    )

    return adj_norm


if __name__ == "__main__":

    # Standalone architecture test
    torch.manual_seed(0)

    batch_size = 8
    num_channels = 9
    samples = 250

    # Fake EEG
    x = torch.randn(
        batch_size,
        num_channels,
        samples,
    )

    # Simple fully connected test graph
    adj = torch.ones(
        num_channels,
        num_channels,
    ) - torch.eye(num_channels)

    adj_norm = normalize_adjacency(adj)

    model = SSVEPGNN(
        num_channels=9,
        num_classes=40,
    )

    logits = model(
        x,
        adj_norm,
    )

    print("Input shape :", x.shape)
    print("Adj shape   :", adj_norm.shape)
    print("Output shape:", logits.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters  :", parameters)
