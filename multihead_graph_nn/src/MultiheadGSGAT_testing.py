import torch
from MultiheadGSGAT import MultiHeadGSGAT
from torch_geometric.data import Data
from graph_constructor_heterogenous import GraphHeterogenous  
import json

# Load the trained model
device = torch.device("cpu")

data_saver = {
    "predicted_wire": [],
    "wire_truth": [],
    "predicted_terminal": [],
    "terminal_truth": [],
    "predicted_action":[],
    "action_truth":[]
}

for d in range(70):
    print(f"SAMPLE_{d}")
    # Load a new sample
    vision_data = f"../synthetic_testing_data/4_class_testing/vision_test/test_sample_{d}.json"
    llm_data = f"../synthetic_testing_data/4_class_testing/llm_test/test_sample_{d}.json"
    label_data = f"../synthetic_testing_data/4_class_testing/labels_test/test_sample_{d}.json"

    # Create a graph instance
    graph = GraphHeterogenous(vision_in=vision_data, llm_in=llm_data, label_in=label_data)
    graph.gen_encodings()
    
    wire_encodings = graph.get_wire_encodings()
    wire_positions = graph.get_wire_positions()
    terminal_encodings = graph.get_terminal_encodings()
    terminal_positions = graph.get_terminal_positions()
    goal_encodings = graph.get_goal_encodings()

    # Convert to PyG Data object
    x = torch.cat([wire_encodings, terminal_encodings, goal_encodings.unsqueeze(0)], dim=0)
    edge_index = graph.get_edge_index()
    edge_attr = graph.get_edge_attr()

    y_wire, y_terminal, y_action = graph.get_labels()  # Wire ID, terminal ID, action (one-hot)
    wire_mask = graph.wire_mask
    terminal_mask = graph.terminal_mask

    data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_wire=y_wire.unsqueeze(0),
            y_terminal=y_terminal.unsqueeze(0),
            y_action=y_action.unsqueeze(0),
            wire_mask=graph.wire_mask,
            terminal_mask=graph.terminal_mask
        )
    data = data.to(device)
    print(f"Y ACTION {data.y_action}")

    checkpoint = torch.load("MultiHead_GSGAT_4class.pth", map_location="cpu")
    model = MultiHeadGSGAT(data.x.shape[1], edge_feat_dim=1, hidden_dim=64, num_actions=4).to(device)

    # # Load only matching parameters
    # model_dict = model.state_dict()
    # # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    # # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # Initialize the new class weights randomly
    with torch.no_grad():
        wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
        terminal_mask_idx = data.terminal_mask.nonzero(as_tuple=False).squeeze()
        # model.action_head.weight[-1] = torch.randn_like(model.action_head.weight[0])
        # model.action_head.bias[-1] = torch.tensor(0.0)  # Or another initialization strategy
        model.load_state_dict(torch.load("MultiHead_GSGAT_4class.pth", map_location=device))
        model.to(device)
        model.eval()  # Set model to evaluation mode
        # Run the model on the new sample
        # with torch.no_grad():
        wire_logits, terminal_logits, action_logits = model(
                data.x.float(), wire_mask, terminal_mask, data.edge_index, data.edge_attr, data.batch
            )
        wire_label = torch.tensor(data.y_wire.item()).to(device)
        terminal_label = torch.tensor(data.y_terminal.item()).to(device)

        action_label = torch.tensor(data.y_action).float().to(device)
        action_label = action_label.argmax(dim=1).long()
        print(action_label)
        
        # Predictions
        wire_pred_global = wire_mask_idx[wire_logits.argmax().item()].item()
        data_saver["predicted_wire"].append(wire_pred_global)
        wire_label_global = wire_mask_idx[wire_label.item()].item()
        data_saver["wire_truth"].append(wire_label_global)
        terminal_pred_global = terminal_mask_idx[terminal_logits.argmax().item()].item()
        data_saver["predicted_terminal"].append(terminal_pred_global)
        terminal_label_global = terminal_mask_idx[terminal_label.item()].item()
        data_saver["terminal_truth"].append(terminal_label_global)
        predicted_action = action_logits.argmax().item()
        data_saver["predicted_action"].append(predicted_action)
        true_action = action_label.item()
        data_saver["action_truth"].append(true_action)
        print(f"True Wire: {wire_label_global}")
        print(f"Predicted Wire: {wire_pred_global}")
        print(f"True terminal: {terminal_label_global}")
        print(f"Predicted terminal: {terminal_pred_global}")
        print(f"True Action: {true_action}")
        print(f"Predicted Action: {predicted_action}")

with open("../docs/MultiHEad_testing.json", "w") as file:
    json.dump(data_saver, file)