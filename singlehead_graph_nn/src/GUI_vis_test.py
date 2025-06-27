import json
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

# Load the saved graph JSON
with open("../run_data/graph_output/inference_graph.json", "r") as f:
    data = json.load(f)

# Reconstruct the NetworkX graph
G = json_graph.node_link_graph(data)
node_labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}

# Basic layout and visualization
pos = nx.kamada_kawai_layout(G)  # or use nx.kamada_kawai_layout(G) for prettier layout

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", edge_color="gray", font_weight="bold")
plt.title("Graph from Inference JSON")
plt.show()
