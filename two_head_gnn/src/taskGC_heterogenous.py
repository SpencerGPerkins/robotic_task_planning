import torch
import networkx as nx 
import json
import numpy as np
from pos_encoding import SpatialPositionalEncoding

"""
Modeling notation:

G = (V, E) where V = V_w union V_t union V_g and E = E_1 union E_2
E_1 = {(n,m) | (n in V and m in V_g) or (n in V_g and m in V)
E_2 = {(n,m) | (n in V_w and m in V_t) or (n in V_t and m in V_w)}

phi_w, phi_t, phi_g are corresponding feature vectors
 
"""
class GraphHeterogenous:
    def __init__(self, action_primitives, goal_states, vision_in, llm_in, label_in=None ):
        self.actions = action_primitives # Possible Predicted actions robot prims
        self.goal_states = goal_states # Possible goal states V_g (insert: terminal state for V_w; terminal state for V_t)
        self.colors = ["red", "yellow", "blue", "green", "black", "white"] # Possible colors for V_w
        self.pos_encoder = SpatialPositionalEncoding(20, in_dim=2) # Specify in_dim based on coordinate dimensions
        
        # Retrieve LLM Data
        with open(llm_in, 'r') as llm_file:
            llm_data = json.load(llm_file)
            
        color, _ = llm_data["target_wire"].split("_")
        _, terminal_number = llm_data["target_terminal"].split("_")

        self.target_info = {
            "wire_color": color,
            "wire_id": self.colors.index(color),  # Index of target wire's color within self.colors
            "terminal_id": int(terminal_number),
            "terminal_name": llm_data["target_terminal"],
            "goal": llm_data["goal"]
        }      
          
        # Retrieve Vision system data
        with open(vision_in, 'r') as vision_file:
            vision_data = json.load(vision_file)
        
        self.wire_dict = []
        for idx, wire in enumerate(vision_data["wires"]): # Iterate through detected wires for target wire
            if wire["color"] == self.target_info["wire_color"]:
                dict_entry = {
                    "id": idx,
                    "color": wire["color"],
                    "coords": wire["coordinates"]
                }
                self.wire_dict.append(dict_entry)
                print(f"Wire {idx} processed...")
            else:
                continue
            
        self.terminal_dict = {
            "id": self.target_info["terminal_id"],
            "name": self.target_info["terminal_name"],
            "coords": vision_data["terminals"][self.target_info["terminal_name"]]["coordinates"]       
        }   
        
        # Normalize object coordinates for positional encoding
        all_wire_coords = [wire["coords"] for wire in self.wire_dict]   
        all_coords = np.array(all_wire_coords + [self.terminal_dict["coords"]])
        norm_coords = self.normalize(all_coords)
        norm_wire_coords = norm_coords[:len(all_wire_coords)]
        norm_terminal_coords = norm_coords[len(all_wire_coords)] # Should only be last index of norm_coords
        
        for i, norm_coord in enumerate(norm_wire_coords):
            self.wire_dict[i]["normalized_coords"] = norm_coord
        self.terminal_dict["normalized_coords"] = norm_terminal_coords
        
        if label_in:
            with open(label_in, 'r') as label_file:
                label_data = json.load(label_file)

            wire_color = label_data["target_wire"]["color"]
            wire_coords = label_data["target_wire"]["coordinates"]
            tar_num = int(label_data["target_terminal"]["name"].split("_")[1])
            # Find the matching wire in self.wire_dict
            # for wire in self.wire_dict:
            #     print(f"THIS:{wire['coords']}")
            #     print(type(wire["coords"]))
            #     print(type(wire_coords))
            #     print("Comparing:", wire["coords"], "vs", wire_coords)

            #     print(self.match_coords(wire["coords"], wire_coords))
            matched_wire = next(
                (wire for wire in self.wire_dict if self.match_coords(wire["coords"], wire_coords)),
                None
            )

            if matched_wire:
                print("\nMatched target wire from label found in detected wires...\n")
            else:
                raise ValueError(f"No matching wire found for coordinates: {wire_coords}")
            
            self.label_info = {
                "wire_color": wire_color,
                "wire_color_idx": self.colors.index(wire_color),  # index in self.detected_wires
                "wire_coords": label_data["target_wire"]["coordinates"],
                "wire_id": matched_wire["id"],

                "terminal_id": tar_num,
                "terminal_coords": label_data["target_terminal"]["coordinates"],

                "action": label_data["correct_action"],
                "action_one_hot": self.one_hot_encode(label_data["correct_action"], self.actions)
            }

    def gen_encodings(self):
        """ Generate graph encodings """
        self.node_feature_encoding()
        self.edge_index_adj_matrix()
        self.edge_feature_encoding()
        self.create_node_masks()
    
    def match_coords(self, a, b, tol=1e-5):
        try:
            a_arr = np.array(a, dtype=np.float64)
            b_arr = np.array(b, dtype=np.float64)
            return np.allclose(a_arr, b_arr, atol=tol)
        except Exception as e:
            print(f"Coord matching failed for {a} vs {b} — {e}")
            return False  
    
    def normalize(self, values):   
        # Compute normalization stats
        mean = values.mean(axis=0)
        std = values.std(axis=0) + 1e-8  
        # Normalize all coords
        norm_coords = (values - mean) / std
        return norm_coords


    def one_hot_encode(self, value, categories):
        """ One-hot Encode Categorical Data"""
        encoding = [0] * len(categories)
        encoding[categories.index(value)] = 1
        return encoding
    
    def euclidean_distance(self, pos1, pos2):
        if type(pos1) == list and type(pos2) == list:
            pos1 = torch.tensor(pos1, dtype=torch.float32) # Convert list to tensor
            pos2 = torch.tensor(pos2, dtype=torch.float32) # Convert list to tensor
            return torch.norm(pos1 - pos2, p=2).item()
        else:
            return torch.norm(pos1 - pos2, p=2).item()
        
    def node_feature_encoding(self):
        X_wires = []
        wire_coords = [np.array(wire["normalized_coords"]) for wire in self.wire_dict]
        coords_list = wire_coords + [self.terminal_dict["normalized_coords"]]
        coords = np.stack(coords_list) # Should be shape: (batchsize=1, N, 2)
        coords = coords[np.newaxis, :, :] # Maintain batch size axis for now in case batching later
        positions = self.pos_encoder(torch.tensor(coords))
        wire_positions = positions[:,:len(self.wire_dict)].squeeze()
        terminal_positions = positions[:,len(self.wire_dict):].squeeze()
        for w, wire in enumerate(self.wire_dict):
            if wire["color"] == self.target_info["wire_color"]:
                if wire_positions.dim() == 1:
                    f = [1., 0.]
                    f.extend(wire_positions.tolist())
                else:
                    f = [1., 0.]
                    f.extend(wire_positions[w,:].tolist())
                X_wires.append(f)
        self.X_wires = X_wires
        
        # Terminal features
        f_term = [0., 1.]
        f_term.extend(terminal_positions.tolist())
        self.X_terminals = f_term
        
    def edge_index_adj_matrix(self):
        num_w = len(self.wire_dict)
        num_t = 1
        edge_index = []

        for i in range(num_w):
            for j in range(num_t):
                edge_index.append([i, num_w + j])
                edge_index.append([num_w + j, i])

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.adj_matrix = torch.zeros(num_w + num_t, num_w + num_t)
        for src, dst in edge_index:
            self.adj_matrix[src, dst] = 1 
                       
    def edge_feature_encoding(self):
        edge_features = []
        distances = []
        num_w, num_t = len(self.wire_dict), len(self.terminal_dict)

        for i in range(num_w):
            dist = self.euclidean_distance(torch.tensor(self.wire_dict[i]["normalized_coords"]), torch.tensor(self.terminal_dict["normalized_coords"]))
            distances.append(dist)
        
        min_dist = min(distances)
        max_dist = max(distances)
        denom = max_dist - min_dist if max_dist > min_dist else 1e-6 # No division by zero
        
        normed = [1 - ((d - min_dist) / denom) for d in distances]

        for val in normed:
            edge_features.append([val]) # Wire to terminal
            edge_features.append([val]) # Terminal to wire

        self.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
    def create_node_masks(self):
        num_w = len(self.wire_dict)
        num_t = 1
        total_nodes = num_w + num_t 

        self.wire_mask = torch.zeros(total_nodes, dtype=torch.bool)
        self.terminal_mask = torch.zeros(total_nodes, dtype=torch.bool)

        self.wire_mask[:num_w] = True
        self.terminal_mask[num_w:num_w + num_t] = True
        
        

    def get_wire_encodings(self):
        return torch.tensor(self.X_wires)

    def get_terminal_encodings(self):
        return torch.tensor(self.X_terminals)

    def get_goal_encodings(self):
        return torch.tensor(self.X_goal)

    def get_edge_index(self):
        print(f"In GC: {len(self.edge_index)}")
        return self.edge_index

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_labels(self):
        return (torch.tensor([self.label_info["wire_id"]]),
                torch.tensor(self.label_info["action_one_hot"]))

    def get_wire_positions(self):
        return torch.tensor([wire["normalized_coords"] for wire in self.wire_dict])

    def get_terminal_positions(self):
        return torch.tensor([terminal["normalized_coords"] for terminal in self.terminal_dict])


    def get_edge_attr(self):
        return self.edge_attr

    def test_mod(self):
        return self.edge_index_adj_matrix()
    
