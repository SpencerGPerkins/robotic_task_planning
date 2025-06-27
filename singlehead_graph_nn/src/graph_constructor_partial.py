import torch
import networkx as nx 
# import matplotlib.pyplot as plt
import json
# from graph_visualization import visualize_graph


"""
Modeling notation:

G = (V, E) where V = V_w union V_t union V_g and E = E_1 union E_2
E_1 = {(n,m) | (n in V and m in V_g) or (n in V_g and m in V)
E_2 = {(n,m) | (n in V_w and m in V_t) or (n in V_t and m in V_w)}

phi_w, phi_t, phi_g are corresponding feature vectors
 
"""

class GraphHeterogenous:
    def __init__(self, vision_in, llm_in, label_in=None):
        self.states = ["on_table", "held", "inserted", "empty", "locked"] # Possible node states for V_w or V_t
        self.actions = ["pick", "insert", "lock", "putdown", "wait"] # Possible Predicted actions (robot prims + wait for non-action)
        self.goal_states = ["insert", "lock"] # Possible goal states V_g (insert: terminal state for V_w; terminal state for V_t)
        self.colors = ["red", "yellow", "blue", "green", "black", "white"] # Possible colors for V_w
        
        # Retrieve Vision system data
        with open(vision_in, 'r') as vision_file:
            vision_data = json.load(vision_file)

        self.detected_wires = [self.colors.index(wire["color"]) for wire in vision_data["wires"]] # Index for wires defined by color (V_w)
        self.wire_positions = [wire["coordinates"] for wire in vision_data["wires"]] # Wire coordinate positions
        self.wire_states = [wire["state"] for wire in vision_data["wires"]] # Wire Current State
        
        self.terminals, self.terminal_positions, self.terminal_states = [], [], [] 
        for terminal, data in vision_data["terminals"].items():
            _, t_num = terminal.split("_") # Find terminal number (identifier)
            self.terminals.append(t_num) #V_t
            self.terminal_positions.append(data["coordinates"]) # Terminal coordinate positions
            self.terminal_states.append(data["state"]) # Terminal current state
            
        # Retrieve LLM Data
        with open(llm_in, 'r') as llm_file:
            llm_data = json.load(llm_file)

        color, _ = llm_data["target_wire"].split("_") # Target wire color
        self.target_wire = self.colors.index(color) # Define target wire by self.colors index (align with self.detected_wires)

        _, number = llm_data["target_terminal"].split("_") # Target terminal
        self.target_terminal = number # Define target terminal by terminal number (align with self.terminals)
        self.target_goal_position = vision_data["terminals"][f"terminal_{self.target_terminal}"]["coordinates"] # Retrieve target terminal's coordinates

        self.goal = llm_data["goal"] # Retrieve LLM defined task goal for target wire / terminal
                
        # Retrieve Labels (TRAINING)
        if label_in:
            with open(label_in, 'r') as label_file:
                label_data = json.load(label_file)
            self.tar_wire_color = label_data["target_wire"]["color"] # Label (ground truth) wire Color
            self.tar_wire_clr_idx = self.colors.index(self.tar_wire_color) # self.color index of GT wire
            self.tar_wire_coords = label_data["target_wire"]["coordinates"] # GT label, coordinates to differentiate like-colored wires

            _, tar_num = label_data["target_terminal"]["name"].split("_") # Label terminal number (GT)
            self.tar_terminal = int(tar_num)
            self.tar_terminal_coords = label_data["target_terminal"]["coordinates"] # GT terminal position
            
            self.y_action = self.one_hot_encode(label_data["correct_action"], self.actions) # GT Action encoding

        # Node feature vectors (phi_w, phi_t, phi_g)
        self.X_wires, self.X_terminals, self.X_goal = {}, {}, {} 
        
        # Define graph connections (E)
        self.edge_index, self.adj_matrix, self.edge_attr = None, None, None
        
    def gen_encodings(self):
        """ Generate graph encodings """
        self.node_feature_encoding()
        self.edge_index_adj_matrix()
        self.edge_feature_encoding()
        
    def one_hot_encode(self, value, categories):
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
        for w, wire in enumerate(self.detected_wires):
            print(wire)
            print(self.target_wire)
            f = [1., 0.]
            f.append(5. if wire == self.target_wire else 1.)
            f.extend([float(x) for x in self.one_hot_encode(self.wire_states[w], self.states)])
            f.extend([float(coord) for coord in self.wire_positions[w]])
            X_wires.append(f)
        self.X_wires = X_wires

        X_terminals = []
        for t, terminal in enumerate(self.terminals):
            f = [0., 1.]
            f.append(5. if terminal == self.target_terminal else 1.)
            f.extend([float(x) for x in self.one_hot_encode(self.terminal_states[t], self.states)])
            f.extend([float(coord) for coord in self.terminal_positions[t]])
            X_terminals.append(f)
        self.X_terminals = X_terminals

        self.X_goal = self.one_hot_encode(self.goal, self.goal_states)
        self.X_goal += [0, 0, 0, 0, 0, 0, 0 ,0]
        
    def edge_index_adj_matrix(self):
        num_w, num_t = len(self.detected_wires), len(self.terminals)
        goal_idx = num_w + num_t
        edge_index = []

        for i in range(num_w):
            for j in range(num_t):
                edge_index.append([i, num_w + j])
                edge_index.append([num_w + j, i])

        for n in range(num_w + num_t):
            edge_index.append([n, goal_idx])
            edge_index.append([goal_idx, n])

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.adj_matrix = torch.zeros(goal_idx + 1, goal_idx + 1)
        for src, dst in edge_index:
            self.adj_matrix[src, dst] = 1

    def edge_feature_encoding(self):
        edge_features = []
        distances = []
        num_w, num_t = len(self.detected_wires), len(self.terminals)

        for i in range(num_w):
            for j in range(num_t):
                dist = self.euclidean_distance(self.wire_positions[i], self.terminal_positions[j])
                distances.append(dist)
        
        min_dist = min(distances)
        max_dist = max(distances)
        denom = max_dist - min_dist if max_dist > min_dist else 1e-6 # No division by zero
        
        normed = [1 - ((d - min_dist) / denom) for d in distances]
        
        for val in normed:
            print(f"normalized distances: {val}")
            edge_features.append([val]) # Wire to terminal
            edge_features.append([val]) # Terminal to wire

        for n in range(num_w):
            print(f"wires: {n}")
            is_target = 1.0 if self.detected_wires[n] == self.target_wire else 0.0
            print(len(edge_features))
            edge_features.append([is_target])
            edge_features.append([is_target])
        for n in range(num_t):
            print(f"terminals: {n}")
            is_target = 1.0 if self.terminals[n] == self.target_terminal else 0.0
            print(edge_features)
            edge_features.append([is_target])
            edge_features.append([is_target])

        self.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
    def get_wire_encodings(self):
        return torch.tensor(self.X_wires)
    
    def get_terminal_encodings(self):
        return torch.tensor(self.X_terminals)
    
    def get_goal_encodings(self):
        return torch.tensor(self.X_goal)
    
    def get_edge_index(self):
        return self.edge_index

    def get_adj_matrix(self):
        return self.adj_matrix
    
    def get_labels(self):
        return torch.tensor(self.y_action)
    
    def get_wire_positions(self):
        return torch.tensor(self.wire_positions)

    def get_terminal_positions(self):
        return torch.tensor(self.terminal_positions)

    def get_edge_attr(self):
        return self.edge_attr      
           
    def test_mod(self):
        n_nodes = self.edge_index_adj_matrix()
        return n_nodes
