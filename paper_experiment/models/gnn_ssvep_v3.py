import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        x = torch.matmul(adj_norm, x)
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


class SSVEPGNNv3(nn.Module):
    """
    Frequency-aware GNN.

    Input:
        EEG [B, 9, T]

    Output:
        40-class logits [B, 40]

    The frequency branch is an analytic Fourier representation.
    It is NOT a CCA classifier and does not use class labels.
    """

    def __init__(
        self,
        num_channels=9,
        num_classes=40,
        graph_dim=64,
        dropout=0.2,
        sampling_rate=250,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate

        # ---------------------------------------------------------
        # Learned temporal branch
        # ---------------------------------------------------------

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

        self.temporal_pool = nn.AdaptiveAvgPool1d(16)

        # ---------------------------------------------------------
        # Frequency branch
        # ---------------------------------------------------------
        #
        # Frequencies are fixed by the Benchmark paradigm:
        # 8.0 to 15.8 Hz in 0.2 Hz steps.
        #
        # We calculate FFT magnitudes from the EEG itself.
        # No target labels are used.
        #

        self.freq_min = 8.0
        self.freq_max = 15.8
        self.freq_step = 0.2

        freq_count = int(
            round(
                (self.freq_max - self.freq_min)
                / self.freq_step
            )
        ) + 1

        self.num_frequency_bins = freq_count

        # Project spectral representation into graph feature space
        self.frequency_projection = nn.Sequential(
            nn.Linear(
                freq_count,
                32,
            ),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Combine temporal + frequency representations
        # ---------------------------------------------------------

        combined_dim = (
            96 * 16
            + 32
        )

        self.node_projection = nn.Sequential(
            nn.Linear(
                combined_dim,
                graph_dim,
            ),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # GNN
        # ---------------------------------------------------------

        self.gcn_input = GraphConv(
            graph_dim,
            graph_dim,
        )

        self.gcn_block1 = ResidualGCNBlock(
            graph_dim,
            dropout,
        )

        self.gcn_block2 = ResidualGCNBlock(
            graph_dim,
            dropout,
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

    def compute_frequency_features(self, x):
        """
        Compute normalized FFT magnitude in the SSVEP
        frequency range.

        Input:
            [B*N, 1, T]

        Output:
            [B*N, number_of_frequency_bins]
        """

        # Remove channel dimension
        signal = x[:, 0, :]

        T = signal.shape[-1]

        # Remove DC component
        signal = signal - signal.mean(
            dim=-1,
            keepdim=True,
        )

        # FFT
        spectrum = torch.fft.rfft(
            signal,
            dim=-1,
        )

        magnitude = torch.abs(spectrum)

        # FFT frequency axis
        frequencies = torch.fft.rfftfreq(
            T,
            d=1.0 / self.sampling_rate,
        ).to(x.device)

        # Extract the SSVEP frequency range
        mask = (
            (frequencies >= self.freq_min)
            &
            (frequencies <= self.freq_max)
        )

        magnitude = magnitude[:, mask]

        # Interpolate to exactly the 40 target frequencies.
        target_frequencies = torch.arange(
            self.freq_min,
            self.freq_max + 0.001,
            self.freq_step,
            device=x.device,
        )

        # Linear interpolation across the FFT spectrum
        # using nearest FFT bins.
        indices = torch.argmin(
            torch.abs(
                frequencies[mask].unsqueeze(0)
                - target_frequencies.unsqueeze(1)
            ),
            dim=1,
        )

        magnitude = magnitude[:, indices]

        # Log compression
        magnitude = torch.log1p(
            magnitude
        )

        # Per-trial normalization
        magnitude = (
            magnitude
            - magnitude.mean(
                dim=-1,
                keepdim=True,
            )
        )

        magnitude = magnitude / (
            magnitude.std(
                dim=-1,
                keepdim=True,
            ) + 1e-6
        )

        return magnitude

    def forward(self, x, adj_norm):

        B, N, T = x.shape

        assert N == self.num_channels

        # ---------------------------------------------------------
        # Process each electrode independently
        # ---------------------------------------------------------

        x = x.reshape(
            B * N,
            1,
            T,
        )

        # ---------------------------------------------------------
        # Learned temporal representation
        # ---------------------------------------------------------

        short = self.temporal_short(x)
        medium = self.temporal_medium(x)
        long = self.temporal_long(x)

        temporal = torch.cat(
            [
                short,
                medium,
                long,
            ],
            dim=1,
        )

        temporal = self.temporal_fusion(
            temporal
        )

        temporal = self.temporal_pool(
            temporal
        )

        temporal = temporal.flatten(
            start_dim=1
        )

        # ---------------------------------------------------------
        # Frequency representation
        # ---------------------------------------------------------

        frequency = self.compute_frequency_features(
            x
        )

        frequency = self.frequency_projection(
            frequency
        )

        # ---------------------------------------------------------
        # Combine
        # ---------------------------------------------------------

        features = torch.cat(
            [
                temporal,
                frequency,
            ],
            dim=1,
        )

        x = self.node_projection(
            features
        )

        x = x.reshape(
            B,
            N,
            -1,
        )

        # ---------------------------------------------------------
        # Graph processing
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
        # Electrode attention
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
        # 40-class prediction
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

    degree = adj_hat.sum(
        dim=1
    )

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5,
    )

    degree_inv_sqrt[
        torch.isinf(
            degree_inv_sqrt
        )
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

    adj = (
        torch.ones(9, 9)
        - torch.eye(9)
    )

    adj_norm = normalize_adjacency(
        adj
    )

    model = SSVEPGNNv3()

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
    print(
        "Frequency bins:",
        model.num_frequency_bins,
    )
