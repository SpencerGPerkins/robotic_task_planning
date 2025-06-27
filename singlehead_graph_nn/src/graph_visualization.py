import networkx as nx 
import matplotlib.pyplot as plt
import datetime
import os
from graph_constructor_partial import GraphHeterogenous


# month = datetime.datetime.now().month
# day = datetime.datetime.now().day
# year = datetime.datetime.now().year
# hour = datetime.datetime.now().hour
# minute = datetime.datetime.now().minute


# def visualize_graph(graph):
#     global month, day, year, hour, minute

#     G = nx.Graph()
    
#     # Initialize node labels and colors
#     node_labels = {}
#     colors = []

#     # Add wire nodes
#     for w, wire in enumerate(graph.detected_wires):
#         G.add_node(w, pos=graph.wire_positions[w])
#         node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
        
#         # Color based on condition
#         colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

#     print(f"Number of terminals: {len(graph.terminals)}")
#     print(f"Number of terminal features: {len(graph.X_terminals)}")
#     # Add terminal nodes
#     wire_count = len(graph.detected_wires)  # Number of wire nodes
#     for t, terminal in enumerate(graph.terminals):
#         t_idx = wire_count + t  # Ensure terminal nodes have unique indices
#         G.add_node(t_idx, pos=graph.terminal_positions[t])
#         node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"  # Fixed indexing
#         colors.append("blue")

#     # Add edges
#     edges = graph.get_edge_index().t().tolist()
#     for src, tgt in edges:
#         G.add_edge(src, tgt)

#     # Compute layout
#     pos = nx.kamada_kawai_layout(G)


#     # Debugging: Check labels
#     print(node_labels)
    
#     plt.figure(figsize=(20, 17))
#     nx.draw(G, pos, node_color=colors, with_labels=True, labels=node_labels, node_size=4500, font_size=8, width=3.0)
#     dir = f"../docs/figures/{year}_{month}_{day}/"
#     os.makedirs(dir, exist_ok=True)
#     save_path = os.path.join(dir, f"{hour}{minute}.png")
#     plt.savefig(save_path)
    
#     plt.show()
    
# vision_in = "../run_data/vision/vision_to_gnn.json"
# llm_in = "../run_data/llm/llm_to_gnn.json"

# G = GraphCategorical(vision_in, llm_in)
# G.gen_encodings()
# print(G)
# visualize_graph(G)

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

    # Add wire nodes
    for w, wire in enumerate(graph.detected_wires):
        G.add_node(w, pos=tuple(graph.wire_positions[w]))
        node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
        colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

    # Add terminal nodes
    wire_count = len(graph.detected_wires)
    for t, terminal in enumerate(graph.terminals):
        t_idx = wire_count + t
        G.add_node(t_idx, pos=tuple(graph.terminal_positions[t]))
        node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"
        colors.append("blue")

    # Add goal node
    total_nodes = wire_count + len(graph.terminals)
    goal_idx = total_nodes
    G.add_node(goal_idx, pos=(0, 0))  # Assign dummy pos — layout will override
    node_labels[goal_idx] = f"Goal\n{graph.goal}"
    colors.append("green")

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

    # Debugging: Check labels
    print(node_labels)

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

    
vision_in = "../run_data/vision/vision_to_gnn.json"
llm_in = "../run_data/llm/llm_to_gnn.json"

G = GraphHeterogenous(vision_in, llm_in)
G.gen_encodings()
print(G.get_edge_index)
print(G)
visualize_graph(G)
print(G.get_edge_index())
# G.verify_edge_features()

