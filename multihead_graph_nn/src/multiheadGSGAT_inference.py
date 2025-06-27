import torch
from MultiheadGSGAT import MultiHeadGSGAT
from torch_geometric.data import Data
from graph_constructor_heterogenous import GraphHeterogenous  
import json
import networkx as nx
from networkx.readwrite import json_graph


def run_inference():

    # Load the trained model
    device = torch.device("cpu")

    possible_actions = ["pick", "insert", "lock", "putdown", "wait"]

    # Save data for history and next action
    data_saver = {
        "predicted_action":[],
        "previous_actions":[]
    }
    with open("../run_data/action_executions.json", "r") as in_file:
        history = json.load(in_file)

    data_saver["previous_actions"] = history["previous_actions"]

    # Load a new sample
    vision_data = f"../run_data/vision/sim_vis_to_gnn.json"
    llm_data = f"../run_data/llm/llm_to_gnn.json"


    # Create a graph instance
    graph = GraphHeterogenous(vision_in=vision_data, llm_in=llm_data)
    graph.gen_encodings()
    
    possible_wires = graph.wire_dict
    possible_terminals = graph.terminal_dict
    
#     # # Initialize node labels and colors
#     node_labels = {}
#     colors = []
#     wire_count = len(graph.wire_dict)
#     G = nx.Graph()
#     # Add wire nodes
#     for w, wire in enumerate(graph.wire_dict):
#         G.add_node(w, pos=graph.wire_positions[w])
#         node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
#         colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

#     # Add terminal nodes
#     for t, terminal in enumerate(graph.terminals):
#         t_idx = wire_count + t
#         G.add_node(t_idx, pos=graph.terminal_positions[t])
#         node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"
#         colors.append("blue")

#     # Add edges (still using integer indices)
#     edges = graph.get_edge_index().t().tolist()
#     for src, tgt in edges:
#         G.add_edge(src, tgt)

# # Export for GUI or JSON
#     graph_data = json_graph.node_link_data(G)



#     with open("../run_data/graph_output/inference_graph.json", "w") as f:
#         json.dump(graph_data, f)
#         print("saved")

    print(graph.get_wire_encodings().shape)
    print(graph.get_terminal_encodings().shape)
    print(graph.get_goal_encodings().unsqueeze(0).shape)
    # Convert to PyG Data object
    x = torch.cat([graph.get_wire_encodings(), graph.get_terminal_encodings(), graph.get_goal_encodings().unsqueeze(0)], dim=0)
    edge_index = graph.get_edge_index()
           
    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=graph.edge_attr,
                wire_mask=graph.wire_mask,
                terminal_mask=graph.terminal_mask
                )  
    data = data.to(device)


    model = MultiHeadGSGAT(in_dim=data.x.shape[1], edge_feat_dim=1, hidden_dim=64, num_actions=5)

    with torch.no_grad():
        model.load_state_dict(torch.load("MultiHead_GSGAT_5class.pth", map_location=device))
        model.to(device)
        model.eval()  

        wire_logits, terminal_logits, action_logits = model(
            data.x.float(), data.wire_mask, data.terminal_mask, data.edge_index, data.edge_attr, data.batch
        )
        
        print(f"\n\nWire Logits: {wire_logits}")
        print(f"Terminal Logitss: {terminal_logits}")
        print(f"Action Logits: {action_logits}\n\n")
        
        # Convert logits to predicted class
        predicted_wire = wire_logits.argmax(dim=0).cpu().numpy()
        predicted_terminal = terminal_logits.argmax(dim=0).cpu().numpy()
        predicted_action = action_logits.argmax(dim=1).cpu().numpy()
        print(predicted_action)
        print(predicted_wire)
        print(predicted_terminal)

        data_saver["predicted_action"] = [possible_actions[predicted_action[-1]]]
        data_saver["previous_actions"].append(possible_actions[predicted_action[-1]])
        print(f"Predicted Wire: {possible_wires[predicted_wire]['color']}, {possible_wires[predicted_wire]['id']}, {possible_wires[predicted_wire]['coords']}")
        print(f"Predicted Terminal: {possible_terminals[predicted_terminal]['name']}, {possible_terminals[predicted_terminal]['id']}, {possible_terminals[predicted_terminal]['coords']}")
        print(f"Predicted Action: {possible_actions[predicted_action[0]]}")

    with open("../run_data/action_executions.json", "w") as file:
        json.dump(data_saver, file)

run_inference()