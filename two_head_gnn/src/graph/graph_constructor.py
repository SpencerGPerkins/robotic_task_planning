import torch
import numpy as np

from data_process.open_file import open_helper
from data_process.preprocess import (
    parse_target_info,
    extract_wire_nodes,
    extract_terminal_node,
    match_label_to_wire
)
from features.node_features import build_node_features
from features.edge_features import build_edge_index_adj_matrix, edge_feature_encoding
from features.pos_encoding import SpatialPositionalEncoding
from utils.coords import match_coords, normalize
from utils.one_hot import one_hot_encode


class TaskGraphHeterogeneous:
    def __init__(self, action_primitives, vision_path, llm_path, label_path=None):
        """Task-informed Heterogeneous Graph Constructor
        Params:
        -------
        action_primitives:
            list, all possible actions for the robot
        vision_path:
            str, path to vision data json file
        llm_path:
            str, path to LLM data json file
        label_path:
            str, label json file for supervised learning

        ## Generates the graph, use "getting functions" lns 91 ---- to grab data
        """
        
        self.actions = action_primitives 
        self.colors = ["red", "yellow", "blue", "green", "black", "white", "orange"]
        self.positional_encoder = SpatialPositionalEncoding(60, in_dim=3)

        llm_data = open_helper(llm_path)
        vision_data = open_helper(vision_path)
        if label_path:
            label_data = open_helper(label_path)
        
        self.target_info = parse_target_info(llm_data, self.colors)
        self.wire_nodes = extract_wire_nodes(vision_data["wires"], self.target_info)
        self.terminal_node = extract_terminal_node(vision_data["terminals"], self.target_info)

        # Normalize object coordinates for positional encoding
        all_wire_coords = [wire["coordinates"] for wire in self.wire_nodes]
        print(all_wire_coords)
        print(self.terminal_node["coordinates"])
        all_coords = np.array(all_wire_coords + [self.terminal_node["coordinates"]])
        norm_coords = normalize(all_coords)
        norm_wire_coords = norm_coords[:len(all_wire_coords)]
        norm_terminal_coords = norm_coords[len(all_wire_coords)] # Should only be last index of norm_coords
        # Assign normalized coordinates to wire and terminal node dicts
        for i, norm_coord in enumerate(norm_wire_coords):
            self.wire_nodes[i]["normalized_coordinates"] = norm_coord
        self.terminal_node["normalized_coordinates"] = norm_terminal_coords

        # -------Label info: Match ID of wire from wire_nodes 
        # One-hot encode Action label
        self.label_info = match_label_to_wire(label_data, self.wire_nodes, match_coords, self.colors)
        self.label_info["action_one_hot"] = one_hot_encode(self.label_info["action"], self.actions)

        # ----Graph Features----
        self.X_wires, self.X_terminal, self.label_info["local_wire_id"] = build_node_features(
            self.wire_nodes, self.terminal_node, self.positional_encoder, self.target_info, self.label_info
        )

        self.edge_index, self.adj_matrix = build_edge_index_adj_matrix(len(self.wire_nodes), 1) # 1 for 1 terminal
        self.edge_attr, _ = edge_feature_encoding(self.wire_nodes, self.terminal_node, self.euclidean_distance)

        self.create_node_masks()

    def euclidean_distance(self, pos1, pos2):
        pos1 = torch.tensor(pos1, dtype=torch.float32)
        pos2 = torch.tensor(pos2, dtype=torch.float32)
        return torch.norm(pos1 - pos2, p=2).item()
    
    def create_node_masks(self):
        num_w = len(self.wire_nodes)
        num_t = 1
        total_nodes = num_w + num_t 

        self.wire_mask = torch.zeros(total_nodes, dtype=torch.bool)
        self.terminal_mask = torch.zeros(total_nodes, dtype=torch.bool)

        self.wire_mask[:num_w] = True
        self.terminal_mask[num_w:num_w + num_t] = True

    # -------"Getting" functions
    def get_node_features(self):
        return torch.tensor(self.X_wires), torch.tensor(self.X_terminal)

    def get_edge_index(self):
        return self.edge_index

    def get_edge_attr(self):
        return self.edge_attr

    def get_node_masks(self):
        return self.wire_mask, self.terminal_mask

    def get_labels(self):
        return (
            torch.tensor([self.label_info["global_wire_id"]]),
            torch.tensor([self.label_info["local_wire_id"]]),
            torch.tensor(self.label_info["action_one_hot"])
        )

    def get_positions(self):
        wire_pos = [wire["normalized_coords"] for wire in self.wire_dict]
        terminal_pos = self.terminal_dict["normalized_coords"]
        return torch.tensor(wire_pos), torch.tensor(terminal_pos)
        





        