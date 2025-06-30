from truncated_normal import TruncatedMVN
import torch
import json
import numpy as np
import random


def euclidean_distance(pos1, pos2):
    return  np.linalg.norm(pos1- pos2)

class SyntheticData:
    
    def __init__(self, on_table_dist, held_dist, terminal_coords):
        self.on_table_dist = on_table_dist
        self.held_dist = held_dist
        self.wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
        self.terminals = [
            "terminal_0", "terminal_1", 
            "terminal_2", "terminal_3", 
            "terminal_4", "terminal_5", 
            "terminal_6", "terminal_7", 
            "terminal_8"
            ]

        self.terminal_coords = terminal_coords

    def generate_files(self, num_samples:int, start_samp_num:int, action_type:str):
        """Generate a sample based on action type"""
        for n in range(num_samples):
            num_samp = n + start_samp_num 
            # vision_dict = {
            #     "wires": [],
            #     "terminals": {},
            #     "sample_label": action_type
            # }

            # llm_dict = {
            #     "goal": action_type,
            #     "target_wire": None,
            #     "target_terminal": None
            # }

            # labels_dict = {
            #     "target_wire": {"ID": 0, "name": None, "color": None, "coordinates":[]},
            #     "target_terminal": {"name": None, "coordinates":[]},
            #     "correct_action": action_type
            # }
            target_terminal = random.choice(self.terminals)
            target_wire_color = random.choice(self.wire_colors)
        return 0

    def get_wires_terminals(self, target_wire, target_terminal, num_wires):
        wires = []
        terminals = {}
        for w in range(num_wires):
            if w == target_wire["ID"]:
                wire = {
                    "id": target_wire["ID"],
                    "name": target_wire["name"],
                    "state": "held",
                    "coordinates": target_wire["coordinates"]
                }
            else:
                wire = {
                    "id": w,
                    "name": f"{color}_wire",
                    "color": color,
                    "state": "held",
                    "coordinates": coords.tolist()
                }  
            vision_dict["wires"].append(wire)

        for t, term_key in enumerate(self.terminals):
            terminals[term_key] = {"state": "empty", "coordinates": self.terminal_coords[t]}  

        return wires, terminals  
        
    def get_dicts(self, num_wires, sample_action):
        target_terminal = random.choice(self.terminals)
        target_wire_idx = random.randint(0, num_wires-1)
        target_wire_color = random.choice(self.wire_colors)
        vision_dict = {"wires": [], "terminals": {}, "sample_label": None}
        lables_dict = {"target_wire":{}, "target_terminal":{}, "correct_action": sample_action}
        llm_dict = {"goal": sample_action, "target_wire": target_wire_color , "target_terminal": target_terminal}

        # Get Target Wire and Terminal for Insert and Lock label Files
        if sample_action == "insert":
            labels_dict["target_wire"] = {
                "ID": target_wire_idx,
                "name": f"{target_wire_color}_wire",
                "coordinates": self.held_dist.sample(n=1)
            }
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "coordinates": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            vision_dict["wires"], vision_dict["terminals"] = self.get_wires_terminals(lables_dict["target_wire"], labels_dict["target_terminal"], num_wires)

        elif sample_action == "lock":
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "coordinates": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            inserted_wire_coords = target_terminal["coordinates"]
            labels_dict["target_wire"] = {
                "ID": target_wire_idx,
                "name": f"{target_wire_color}_wire",
                "coordinates": [inserted_wire_coords[0]+0.005, inserted_wire_coords[1], inserted_wire_coords[2]]
            }
            vision_dict["wires"], vision_dict["terminals"] = self.get_wires_terminals(lables_dict["target_wire"], labels_dict["target_terminal"], num_wires)
        else:
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "coordinates": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            labels_dict["target_wire"] = {
                "ID": -1, # Arbitrary, we don't know until distance calculation
                "name": f"{target_wire_color}_wire",
                "coordinates": []
            }   
            vision_dict["wires"], vision_dict["terminals"] = self.get_wires_terminals(lables_dict["target_wire"], labels_dict["target_terminal"], num_wires)





        



    def save_vision(self, vision_dict, sample_number):
        pass
    def save_llm(self, llm_dict, sample_number):
        pass
    def save_labels(self, labels_dict, sample_number):
        pass