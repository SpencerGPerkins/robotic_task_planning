import torch
import networkx as nx 
# import matplotlib.pyplot as plt
import json
# from graph_visualization import visualize_graph

class GraphCategorical:
    
    def __init__(self, vision_in, llm_in, label_in=None):
        
        # Predefined data
        terminals = [
            "0", "1", 
            "2", "3", 
            "4", "5",
            "6", "7",
            "8", "9"
        ] 
        self.states = ["on_table", "held", "inserted", "empty", "locked"]
        self.actions = ["pick", "insert", "lock", "putdown"]
        
        # Process input data
        with open(vision_in,'r') as vision_file: 
            """ VISION FILE """
            vision_data = json.load(vision_file)
        self.detected_wires = [wire["color"] for wire in vision_data["wires"]]
        self.wire_positions = [wire["coordinates"] for wire in vision_data["wires"]]
        self.wire_states = [wire["state"] for wire in vision_data["wires"]]
        
        terminals_names = [terminal for terminal in vision_data["terminals"]]
        self.terminals = []
        self.terminal_positions = []
        self.terminal_states  = []
        for t, terminal in enumerate(terminals_names):
            _, t_num = terminal.split("_")
            self.terminals.append(t_num)
            self.terminal_positions.append(vision_data["terminals"][terminal]["coordinates"])
            self.terminal_states.append(vision_data["terminals"][terminal]["state"])
            
        with open(llm_in, 'r') as llm_file:
            """ LLM FILE """
            llm_data = json.load(llm_file)

        color, _ = llm_data["target_wire"].split("_")      
        self.target_wire = color
        _, number = llm_data["target_terminal"].split("_")
        self.target_terminal = number
        self.target_goal_position = vision_data["terminals"][f"terminal_{self.target_terminal}"]["coordinates"]
        
        if label_in:
            with open(label_in, 'r') as label_file:
                label_data = json.load(label_file)
                self.y_action = self.one_hot_encode(label_data["correct_action"], self.actions)
                
                
        # Initialize encodings and graph structure
        self.X_wires = {}
        self.X_terminals = {}
        self.edge_index = None
        self.adj_matrix = None 
        self.edge_features = None 
        
    def gen_encodings(self):
        # Generate encodings and graph
        self.node_feature_encoding()
        self.edge_index_adj_matrix()

        
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
        states = ["on_table", "held", "inserted", "empty", "locked"]
        max_distance = 10e10 # Set initial distance for checking node distances
        wire_distance_weight = 2.5
        terminal_distance_weight = 2.5
        
        # Encode wires
        X_wires = []
        for w, wire in enumerate(self.detected_wires):
            wire_features = [1.,0.]
            if wire == self.target_wire: # Check if wire is potential target
                wire_features.append(5.0)
                distance_from_goal = self.euclidean_distance(self.wire_positions[w], self.target_goal_position)
                if distance_from_goal < max_distance:
                    max_distance = distance_from_goal
                    wire_features.append(wire_distance_weight)
                else:
                    wire_features.append(0.0)
            else:        
                wire_features.append(1.0) # Not target
                wire_features.append(0.0) # Distance is not important
            wire_state_encoding = self.one_hot_encode(self.wire_states[w], self.states) # Encode State
            for encoding in wire_state_encoding:
                wire_features.append(float(encoding))
            X_wires.append(wire_features)
        self.X_wires = X_wires
        
        # Encode terminals
        X_terminals = []
        for t, terminal in enumerate(self.terminals):
            terminal_features = [0.,1.]
            if terminal == self.target_terminal:
                terminal_features.append(5.0) # Target terminal weight
            else:
                terminal_features.append(1.0) 
            terminal_features.append(0.0) # Distance weight set to default
            terminal_state_encoding = self.one_hot_encode(self.terminal_states[t], self.states)
            for e, encoding in enumerate(terminal_state_encoding):
                terminal_features.append(float(encoding))
        
            X_terminals.append(terminal_features)
        self.X_terminals = X_terminals
        
    def edge_index_adj_matrix(self):
        # Combine wire and terminal nodes to one list
        wire_nodes = self.detected_wires
        terminal_nodes = self.terminals
        all_nodes = wire_nodes + terminal_nodes
        num_wirenodes = len(wire_nodes)
        num_termnodes = len(terminal_nodes)
        # Create edge index (only connections from different categories)
        edge_index = [[i, j] for i in range(num_wirenodes) for j in range(num_wirenodes, num_wirenodes + len(terminal_nodes))]
        # Add reverse edges (terminal → wire)
        edge_index += [[j, i] for i in range(num_wirenodes) 
                            for j in range(num_wirenodes, num_wirenodes + len(terminal_nodes))]
        
        # Convert edge_index to tensor
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 
        
        num_nodes = len(wire_nodes) + len(terminal_nodes)
        # Create adjacency matrix A
        self.adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        
    def get_wire_encodings(self):
        return torch.tensor(self.X_wires)
    
    def get_terminal_encodings(self):
        return torch.tensor(self.X_terminals)
    
    def get_edge_index(self):
        return self.edge_index

    def get_adj_matrix(self):
        return self.adj_matrix
    
    def get_labels(self):
        # return torch.tensor(self.y_target_wire), torch.tensor(self.y_target_terminal), torch.tensor(self.y_action)
        return torch.tensor(self.y_action)
        
                
    def test_mod(self):
        n_nodes = self.edge_index_adj_matrix()
        return n_nodes
