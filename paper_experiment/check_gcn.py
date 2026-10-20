import torch
import numpy as np

graph = np.load("paper_experiment/graph_9ch.npz")

A = torch.tensor(graph["adjacency"], dtype=torch.float32)

# Add self-loops for GCN propagation
A_hat = A + torch.eye(A.shape[0])

# Symmetric normalization
degree = A_hat.sum(dim=1)
D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))

A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

print("Adjacency shape:", A.shape)
print("Normalized adjacency shape:", A_norm.shape)
print("Symmetric:", torch.allclose(A_norm, A_norm.T))
print("Finite:", torch.isfinite(A_norm).all().item())
print("Diagonal after self-loop:", A_norm.diag())

# Smoke-test graph propagation
X = torch.randn(9, 32)
W = torch.randn(32, 64)

H = A_norm @ X @ W

print("Input:", X.shape)
print("Output:", H.shape)
print("Output finite:", torch.isfinite(H).all().item())
