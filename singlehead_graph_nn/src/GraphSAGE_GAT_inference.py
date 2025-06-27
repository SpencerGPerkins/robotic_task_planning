import torch
from GraphSAGE_GAT import GraphSAGE_GAT
from torch_geometric.data import Data
from graph_constructor_partial import GraphCategorical  
import json
import networkx as nx
from networkx.readwrite import json_graph


def run_inference():

    # Load the trained model
    device = torch.device("cpu")

    possible_actions = ["pick", "insert", "lock", "putdown"]

    # Save data for history and next action
    data_saver = {
        "predicted_action":[],
        "previous_actions":[]
    }
    with open("../run_data/action_executions.json", "r") as in_file:
        history = json.load(in_file)

    data_saver["previous_actions"] = history["previous_actions"]

    # Load a new sample
    vision_data = f"../run_data/vision/vision_to_gnn.json"
    llm_data = f"../run_data/llm/llm_to_gnn.json"


    # Create a graph instance
    graph = GraphCategorical(vision_in=vision_data, llm_in=llm_data)
    graph.gen_encodings()
    
    
    # # Create Network X graph for GUI visualization
    # G = nx.DiGraph()

    # wire_names = graph.detected_wires
    # terminal_names = graph.terminals
    # all_node_names = wire_names + terminal_names

    # # Add nodes with features (optional)
    # for i, node in enumerate(all_node_names):
    #     if i < len(graph.X_wires):
    #         features = graph.X_wires[i]
    #     else:
    #         features = graph.X_terminals[i - len(graph.X_wires)]
    #     G.add_node(node, features=features)

    # # Add edges from edge_index
    # edge_index = graph.get_edge_index()
    # for src, dst in edge_index.t().tolist():
    #     G.add_edge(all_node_names[src], all_node_names[dst])

    # # Save to JSON for GUI or any format you need
    # graph_data = json_graph.node_link_data(G)
    # G = nx.Graph()
    
    # # Initialize node labels and colors
    node_labels = {}
    colors = []
    wire_count = len(graph.detected_wires)
    G = nx.Graph()
    # Add wire nodes
    for w, wire in enumerate(graph.detected_wires):
        G.add_node(w, pos=graph.wire_positions[w])
        node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
        colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

    # Add terminal nodes
    for t, terminal in enumerate(graph.terminals):
        t_idx = wire_count + t
        G.add_node(t_idx, pos=graph.terminal_positions[t])
        node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"
        colors.append("blue")

    # Add edges (still using integer indices)
    edges = graph.get_edge_index().t().tolist()
    for src, tgt in edges:
        G.add_edge(src, tgt)

# Optional: draw with labels and colors
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos, labels=node_labels, node_color=colors, with_labels=True)

# Export for GUI or JSON
    graph_data = json_graph.node_link_data(G)
    # node_labels = {}
    # colors = []

    # # Add wire nodes
    # for w, wire in enumerate(graph.detected_wires):
    #     G.add_node(w, pos=graph.wire_positions[w])
    #     node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
        
    #     # Color based on condition
    #     colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

    # print(f"Number of terminals: {len(graph.terminals)}")
    # print(f"Number of terminal features: {len(graph.X_terminals)}")
    # # Add terminal nodes
    # wire_count = len(graph.detected_wires)  # Number of wire nodes
    # for t, terminal in enumerate(graph.terminals):
    #     t_idx = wire_count + t  # Ensure terminal nodes have unique indices
    #     G.add_node(t_idx, pos=graph.terminal_positions[t])
    #     node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"  # Fixed indexing
    #     colors.append("blue")

    # # Add edges
    # edges = graph.get_edge_index().t().tolist()
    # for src, tgt in edges:
    #     G.add_edge(src, tgt)
    # graph_data = json_graph.node_link_data(G)



    with open("../run_data/graph_output/inference_graph.json", "w") as f:
        json.dump(graph_data, f)
        print("saved")


    # Convert to PyG Data object
    x = torch.cat([graph.get_wire_encodings(), graph.get_terminal_encodings()], dim=0)
    edge_index = graph.get_edge_index()

    data = Data(x=x, edge_index=edge_index)  
    data = data.to(device)

    checkpoint = torch.load("GraphSAGE_model_weights_4_class.pth", map_location="cpu")
    model = GraphSAGE_GAT(
        in_dim=data.x.shape[1],  # Use the actual feature size
        hidden_dim=16, 
        max_wires=10000, 
        max_terminals=10, 
        num_actions=4
    )

    with torch.no_grad():
        model.load_state_dict(torch.load("GraphSAGE_model_weights_4_class.pth", map_location=device))
        model.to(device)
        model.eval()  

        action_logits = model(
            data.x.float(), data.edge_index, None  
            )
        
        print(action_logits)
        
        # Convert logits to predicted class
        predicted_action = action_logits.argmax(dim=1).cpu().numpy().tolist()

        data_saver["predicted_action"] = [possible_actions[predicted_action[-1]]]
        data_saver["previous_actions"].append(possible_actions[predicted_action[-1]])

        print(f"Predicted Action: {possible_actions[predicted_action[0]]}")

    with open("../run_data/action_executions.json", "w") as file:
        json.dump(data_saver, file)


run_inference()