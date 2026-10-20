import numpy as np
from pathlib import Path

channels = [
    "Pz", "PO3", "PO5", "PO4", "PO6",
    "POz", "O1", "Oz", "O2"
]

coordinates = {
    "Pz":  (180, 0.25556),
    "PO3": (-157, 0.41111),
    "PO5": (-144, 0.44722),
    "PO4": (157, 0.41111),
    "PO6": (144, 0.44722),
    "POz": (180, 0.38333),
    "O1":  (-162, 0.51111),
    "Oz":  (180, 0.51111),
    "O2":  (162, 0.51111),
}

# Convert polar coordinates from the .loc file
xy = []

for ch in channels:
    angle_deg, radius = coordinates[ch]
    angle = np.deg2rad(angle_deg)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    xy.append([x, y])

xy = np.asarray(xy)

# Pairwise Euclidean distances
dist = np.linalg.norm(
    xy[:, None, :] - xy[None, :, :],
    axis=2
)

# k-nearest-neighbor graph
k = 3
adj = np.zeros((len(channels), len(channels)), dtype=np.float32)

for i in range(len(channels)):
    neighbors = np.argsort(dist[i])[1:k+1]
    adj[i, neighbors] = 1.0

# Make graph undirected
adj = np.maximum(adj, adj.T)

# No self loops in stored graph
np.fill_diagonal(adj, 0)

print("Channels:")
for i, ch in enumerate(channels):
    print(f"{i}: {ch}")

print()
print("Coordinates:")
for i, ch in enumerate(channels):
    print(f"{ch}: ({xy[i,0]:.5f}, {xy[i,1]:.5f})")

print()
print("Adjacency:")
print(adj.astype(int))

print()
print("Edges:", int(adj.sum() / 2))
print("Connected:", end=" ")

# Simple BFS connectivity test
visited = {0}
stack = [0]

while stack:
    node = stack.pop()

    for neighbor in np.where(adj[node] > 0)[0]:
        if neighbor not in visited:
            visited.add(int(neighbor))
            stack.append(int(neighbor))

print(len(visited) == len(channels))

out = Path("paper_experiment")
np.savez(
    out / "graph_9ch.npz",
    channels=np.array(channels),
    coordinates=xy,
    adjacency=adj,
    distances=dist
)

print()
print("Saved: paper_experiment/graph_9ch.npz")
