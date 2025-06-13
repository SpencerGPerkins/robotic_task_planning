import networkx as nx 
import matplotlib.pyplot as plt
import datetime
import os
from taskGC_heterogenous import GraphHeterogenous


month = datetime.datetime.now().month
day = datetime.datetime.now().day
year = datetime.datetime.now().year
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute


def visualize_graph(graph):
    global month, day, year, hour, minute

    G = nx.Graph()
    
    # Initialize node labels and colors
    node_labels = {}
    colors = []
    counter = 0
    # Add wire nodes
    for w, wire in enumerate(graph.wire_dict):
        G.add_node(w, pos=tuple(wire["normalized_coords"]))
        node_labels[w] = f"{wire['color'], wire['id']}\n{graph.X_wires[w]}"
        colors.append("green")
        counter += 1

    # Add terminal nodes
    G.add_node(0, pos=tuple(graph.terminal_dict["normalized_coords"]))
    node_labels[counter] = f"{graph.terminal_dict['name']}\n{graph.X_terminals}"
    colors.append("red")


    # # Add edges
    # edges = graph.get_edge_index().t().tolist()
    # for src, tgt in edges:
    #     G.add_edge(src, tgt)
# Add edges with edge feature as label
    edges = graph.get_edge_index().t().tolist()
    edge_labels = {}
    for idx, (src, tgt) in enumerate(edges):
        feature = graph.edge_attr[idx].item() if graph.edge_attr is not None else 0.0
        G.add_edge(src, tgt)
        edge_labels[(src, tgt)] = f"{feature:.2f}"
        

    # Compute layout
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)

    # Ensure all nodes have a position
    for node in G.nodes:
        if node not in pos:
            pos[node] = (0, 0)  # fallback

    # Draw
    plt.figure(figsize=(20, 17))
    nx.draw(G, pos, node_color=colors, with_labels=True,
            labels=node_labels, node_size=4500, font_size=8, width=3.0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    # Save
    dir = f"../docs/figures/{year}_{month}_{day}/"
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"{hour}{minute}.png")
    plt.savefig(save_path)
    plt.show()

    
vision_in = "../synthetic_data/4_class/vision/sample_2.json"
llm_in = "../synthetic_data/4_class/llm/sample_2.json"
label_in = "../synthetic_data/4_class/labels/sample_2.json"

G = GraphHeterogenous(
    ["pick", "insert", "lock"],
    ["insert", "lock"],
    vision_in,
    llm_in,
    label_in=label_in
    )

G.gen_encodings()
print(type(G.X_wires), type(G.X_terminals))
visualize_graph(G)


