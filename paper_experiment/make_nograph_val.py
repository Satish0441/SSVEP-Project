from pathlib import Path

src = Path("paper_experiment/train_gnn_v2_val.py")
dst = Path("paper_experiment/train_gnn_v2_nograph_val.py")

text = src.read_text(encoding="utf-8")

text = text.replace(
    "from models.gnn_ssvep_v2 import SSVEPGNNv2, normalize_adjacency",
    "from models.gnn_ssvep_v2_nograph import SSVEPGNNv2NoGraph"
)

text = text.replace(
    'graph = np.load(GRAPH_FILE)\nadjacency = torch.tensor(graph["adjacency"], dtype=torch.float32)\nadj_norm = normalize_adjacency(adjacency)',
    'adj_norm = None  # Not used by the no-graph model'
)

text = text.replace(
    "model = SSVEPGNNv2(num_channels=9, num_classes=40).to(device)",
    "model = SSVEPGNNv2NoGraph(num_channels=9, num_classes=40).to(device)"
)

text = text.replace(
    'checkpoint_path = Path("paper_experiment/gnn_v2_s1_dev_best.pt")',
    'checkpoint_path = Path("paper_experiment/gnn_v2_nograph_s1_dev_best.pt")'
)

dst.write_text(text, encoding="utf-8")
print(f"Created {dst}")
