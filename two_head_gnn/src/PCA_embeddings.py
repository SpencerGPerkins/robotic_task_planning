import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_process.open_file import open_helper
from features.node_features import build_node_features
from features.pos_encoding import SpatialPositionalEncoding
from graph.graph_constructor import TaskGraphHeterogeneous
import config

base_path = "../../dataset/generated_synthetic_dataset_0807/"
samples = ["sample_0", "sample_1003", "sample_2003", "sample_3003"]
file_map = ["pick", "insert", "lock", "putdown"]

# for sample in samples:
#     graph = TaskGraphHeterogeneous(
#         config.ACTION_PRIMS, 
#         f"{base_path}vision/{sample}.json",
#         f"{base_path}llm/{sample}.json",
#         f"{base_path}labels/{sample}.json"
#     )

#     wire_embeddings, terminal_embeddings = graph.get_node_features()

#     wire_embed = [wire_embed.numpy() for wire_embed in wire_embeddings]
#     terminal_embed = terminal_embeddings.numpy()


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



for sample in samples:
    all_embeddings = []
    labels = []
    graph = TaskGraphHeterogeneous(
        config.ACTION_PRIMS, 
        f"{base_path}vision/{sample}.json",
        f"{base_path}llm/{sample}.json",
        f"{base_path}labels/{sample}.json"
    )

    wire_embeddings, terminal_embeddings = graph.get_node_features()

    # make sure both are 2D arrays
    wire_embed = np.stack([w.numpy() for w in wire_embeddings], axis=0)  # (num_wires, D)

    terminal_embed = terminal_embeddings.numpy()
    if terminal_embed.ndim == 1:   # (D,) -> (1, D)
        terminal_embed = terminal_embed[None, :]

    all_embeddings.append(wire_embed)
    labels.extend(["wire"] * wire_embed.shape[0])

    all_embeddings.append(terminal_embed)
    labels.extend(["terminal"] * terminal_embed.shape[0])

    # stack into one big (N, D) array
    all_embeddings = np.vstack(all_embeddings)

    # ---- PCA ----
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.7)

    plt.legend()
    plt.title("PCA of Wire & Terminal Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
