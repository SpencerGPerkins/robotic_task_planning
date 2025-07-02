import numpy as np
import torch

def build_node_features(wire_list, terminal_dict, pos_encoder, target_info, label_info):
    X_wires = []

    wire_coords = [np.array(wire["normalized_coordinates"]) for wire in wire_list]
    coords_list = wire_coords + [terminal_dict["normalized_coordinates"]]
    coords = np.stack(coords_list) # Should be shape: (batchsize=1, N, 2)
    coords = coords[np.newaxis, :, :] # Maintain batch size axis for now in case batching later, previous 6-dim coords[np.newaxis, :, :]
    positions = pos_encoder(torch.tensor(coords))
    wire_positions = positions[:,:len(wire_list)].squeeze()
    terminal_positions = positions[:,len(wire_list):].squeeze()

    local_wire_id = None
    for w, wire in enumerate(wire_list):
        if wire_positions.dim() == 1:
            f = [1., 0.]
            f.extend(wire_positions.tolist())
        else:
            f = [1., 0.]
            f.extend(wire_positions[w,:].tolist())
        X_wires.append(f)
        if wire["id"] == label_info["global_wire_id"]:
            label_info["local_wire_id"] = len(X_wires)-1  
            local_wire_id = label_info["local_wire_id"]          
    X_wires = X_wires
    
    # Terminal features
    f_term = [0., 1.]
    f_term.extend(terminal_positions.tolist())
    X_terminals = f_term

    return X_wires, X_terminals, local_wire_id
        