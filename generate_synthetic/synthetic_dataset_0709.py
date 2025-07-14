from truncated_normal import TruncatedMVN
import torch
import json
import numpy as np
import random
import os

def euclidean_distance(pos1, pos2):
    return  np.linalg.norm(pos1- pos2)

class SyntheticData:
    def __init__(self, on_table_dist, held_dist, inserted_dist, terminal_coords, config):
        self.on_table_dist = on_table_dist
        self.held_dist = held_dist
        self.inserted_dist = inserted_dist
        self.wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
        self.terminals = [
            "terminal_0", "terminal_1", 
            "terminal_2", "terminal_3", 
            "terminal_4", "terminal_5", 
            "terminal_6", "terminal_7", 
            "terminal_8"
            ]

        self.terminal_coords = terminal_coords
        self.config = config

        self._make_output_dirs()

    def _make_output_dirs(self):
        base = self.config["output_base_path"]
        for sub in ["vision", "llm", "labels"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def generate_files(self, num_samples:int, start_samp_num:int, action_type:str):
        """Generate a sample based on action type
        Params:
        -------
        num_samples: int, total number of samples to generate
        start_samp_num: int, number for starting batch of samples (used for generating consecutive file names)
        action_type: str, the action label for the sample

        Returns:
        -------
        Saves dictionaries as json_files
        """
        parent_directory = self.config["output_base_path"]
        
        for n in range(num_samples):
            sample_number = n + start_samp_num 

            vision, llm, labels = self.get_dicts(action_type)

            # Save dicts to json files
            self.save_vision(vision, sample_number, parent_directory)
            self.save_llm(llm, sample_number, parent_directory)
            self.save_labels(labels, sample_number, parent_directory)

        print(f"All files saved!") 

    def get_wires_terminals(self, target_wire, target_terminal, action_type, num_wires, non_target_colors):
        wires = []
        terminals = {}
        current_color_idx = 0 # Counter for non-color idxs
        for w in range(num_wires): 
            if w == target_wire["ID"] and action_type =="insert":
                wire = {
                    "id": target_wire["ID"],
                    "name": target_wire["name"],
                    "color": target_wire["name"].split("_")[0],
                    "state": "held",
                    "posiiton": target_wire["position"]
                }
            elif w == target_wire["ID"] and action_type == "lock":
                wire = {
                    "id": target_wire["ID"],
                    "name": target_wire["name"],
                    "color": target_wire["name"].split("_")[0],
                    "state": "inserted",
                    "position": target_wire["position"]
                }
            elif w == target_wire["ID"] and action_type == "pick":
                wire = {
                    "id": target_wire["ID"],
                    "name": target_wire["name"],
                    "color": target_wire["name"].split("_")[0],
                    "state": "on_table",
                    "position": target_wire["position"]
                }
            
            else:
                coords = self.on_table_dist.sample(n=1)
                if self.config["generate_one_wire_per_color"] == True:
                    wire = {
                        "id": w,
                        "name": f"{non_target_colors[current_color_idx]}_wire",
                        "color": non_target_colors[current_color_idx],
                        "state": "on_table",
                        "position": coords.flatten().tolist()
                    }
                
                else:
                    color = random.choice(self.wire_colors)
                    wire = {
                        "id": w,
                        "name": f"{color}_wire",
                        "color": color,
                        "state": "on_table",
                        "position": coords.flatten().tolist()
                    }
            wires.append(wire)

            for t, term_key in enumerate(self.terminals):
                if term_key == target_terminal["name"] and action_type == "lock":
                    terminals[term_key] = {"state": "inserted", "position":target_terminal["position"]}
                else:
                    terminals[term_key] = {"state": "empty", "position": self.terminal_coords[t]}
            
            return wires, terminals


    def get_dicts(self, sample_action):

        target_terminal = random.choice(self.terminals)
        target_wire_color = random.choice(self.wire_colors)
        target_wire_idx = self.wire_colors.index(target_wire_color)
        if self.config["generate_one_wire_per_color"] == True:
            num_wires = len(self.wire_colors)
            non_target_colors = [c for c in self.wire_colors if c != target_wire_color]
        else:
            num_wires = random.randint(1, 20)

        # Initialize dicts
        vision_dict = {"wires": [], "terminals": {}, "sample_label": None}
        labels_dict = {"target_wire":{}, "target_terminal": {}, "correct_action": sample_action}
        llm_dict = {"goal": sample_action, "target_wire": f"{target_wire_color}_wire", "target_terminal": target_terminal}

        # Insert sample
        if sample_action == "insert":
            labels_dict["target_wire"] = {
                "ID": target_wire_idx,
                "name": f"{target_wire_color}_wire",
                "position": self.held_dist.sample(n=1).flatten().tolist()
            }
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "position": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            vision_dict["wires"], vision_dict["terminals"] = self.get_wires_terminals(
                labels_dict["target_wire"], labels_dict["target_terminal"], sample_action, num_wires, non_target_colors
            )
            vision_dict["sample_label"] = "insert"

        # Lock sample
        elif sample_action == "lock":
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "position": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            labels_dict["target_wire"] = {
                "ID": target_wire_idx,
                "name": f"{target_wire_color}_wire",
                "position": self.inserted_dist.sample(n=1).flatten().tolist()
            }
            vision_dict["wires"], vision_dict["terminals"] = self.get_wires_terminals(
                labels_dict["target_wire"], labels_dict["target_terminal"], sample_action, num_wires, non_target_colors
            )
            vision_dict["sample_label"] = "lock"
        # Pick sample
        else:
            labels_dict["target_terminal"] = {
                "name": target_terminal,
                "position": self.terminal_coords[self.terminals.index(target_terminal)]
            }
            labels_dict["target_wire"] = {
                "ID": target_wire_idx,
                "name":f"{target_wire_color}_wire",
                "position": self.on_table_dist.sample(n=1).flatten().tolist()
            }
            vision_dict["wire"], vision_dict["terminals"] = self.get_wires_terminals(
                labels_dict["target_wire"], labels_dict["target_terminal"], sample_action, num_wires, non_target_colors
            )

            if self.config["generate_one_wire_per_color"] != True:
                # Find closest target_color wire to target terminal
                potential_target_wire = {"wire_id": -1, "distance_to_term": 1000, "wire_coords": []}
                for w, wire in enumerate(vision_dict["wires"]):
                    if wire["color"] == target_wire_color:
                        distance = euclidean_distance(np.array(wire["position"]), np.array(labels_dict["target_terminal"]["position"]))
                        if distance < potential_target_wire["distance_to_term"]:
                            potential_target_wire["wire_id"] = wire["id"]
                            potential_target_wire["distance_to_term"] = distance
                            potential_target_wire["wire_coords"] = wire["position"]
                        else:
                            continue
                vision_dict["wires"][potential_target_wire["wire_id"]]["color"] = target_wire_color
                vision_dict["wires"][potential_target_wire["wire_id"]]["name"] = f"{target_wire_color}_wire"
                labels_dict["target_wire"]["ID"] = potential_target_wire["wire_id"]
                labels_dict["target_wire"]["position"] = potential_target_wire["wire_coords"]
                vision_dict["sample_label"] = "pick"

        return vision_dict, llm_dict, labels_dict           


    def save_vision(self, vision_dict, sample_number, parent_dir):
        with open(f"{parent_dir}vision/sample_{sample_number}.json", "w") as out_vision:
            json.dump(vision_dict, out_vision)
        print(f"----\nVision sample_{sample_number} saved to {parent_dir}vision/sample_{sample_number}.json")

    def save_llm(self, llm_dict, sample_number, parent_dir):
        with open(f"{parent_dir}llm/sample_{sample_number}.json", "w") as out_llm:
            json.dump(llm_dict, out_llm)
        print(f"llm sample_{sample_number} saved to {parent_dir}llm/sample_{sample_number}.json")

    def save_labels(self, labels_dict, sample_number, parent_dir):
        with open(f"{parent_dir}labels/sample_{sample_number}.json", "w") as out_labels:
            json.dump(labels_dict, out_labels)
        print(f"Labels sample_{sample_number} saved to {parent_dir}labels/sample_{sample_number}.json\n----")       