from truncated_normal import TruncatedMVN
import torch
import json
import numpy as np
import random

class SyntheticData:
    
    def __init__(self, on_table_dist, held_dist, terminal_coords):
        self.on_table_dist = on_table_dist
        self.held_dist = held_dist
        self.wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
        self.terminals = ["terminal_0", "terminal_1", "terminal_2", "terminal_3", "terminal_4", "terminal_5", "terminal_6", "terminal_7", "terminal_8"]
        self.terminal_coords = terminal_coords
        
    def generate_pick_vision(self, num_samples):
        for n in range(num_samples):
            vision_dict = {
                "wires": [],
                "terminals": {}
            }
                 
            num_wires = random.randint(3, 20)
            for w in range(num_wires):
                color = random.choice(self.wire_colors)
                coords = self.on_table_dist.sample(n=1)
                wire = {
                    "id": w,
                    "name": f"{color}_wire",
                    "color": color,
                    "state": "on_table",
                    "coordinates": coords.tolist()
                }
                vision_dict["wires"].append(wire)
            for t, term_key in enumerate(self.terminals):
                vision_dict["terminals"][term_key] = {"state": "empty", "coordinates": self.terminal_coords[t]}
            
            with open(f"../dataset/generated_synthetic/vision/sample_{n}.json", "w") as out_file:
                json.dump(vision_dict, out_file)
            print(f"Sample_{n} saved to ../dataset/generated_synthetic/vision/sample_{n}.json")
                
def get_max_likelihood(matrix):
    mean_vector = torch.mean(matrix, dim=0) # Mean for columns (xyz)
    covariance_mat = torch.cov(matrix.T) # Transpose, cov matrix dims should be (3,3)
    
    return mean_vector, covariance_mat

def process_file(vision_pth, num_files=100):
    table_wirepoints = []
    held_wirepoints = []
    inserted_wirepoints = []
    terminalpoints = []

    for d in range(num_files):
        with open(f"{vision_pth}/sample_{d}.json", "r") as v_in:
            vision_file = json.load(v_in)
        for wire in vision_file["wires"]: # Get all wire coords
            if wire["state"] == "on_table":
                table_wirepoints.append(torch.tensor(wire["coordinates"]))
            elif wire["state"] == "held":
                held_wirepoints.append(torch.tensor(wire["coordinates"]))
            elif wire["state"] == "inserted":
                inserted_wirepoints.append(torch.tensor(wire["coordinates"]))
            else:
                raise ValueError("Unkown state found in wires...")
        term_keys = list(vision_file["terminals"].keys())
        for key in term_keys:
            terminalpoints.append(torch.tensor(vision_file["terminals"][key]["coordinates"]))
            
    return table_wirepoints, held_wirepoints, inserted_wirepoints, terminalpoints



def main():
    wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
    terminals = ["terminal_0", "terminal_1", "terminal_2", "terminal_3", "terminal_4", "terminal_5", "terminal_6", "terminal_7", "terminal_8"]
    
    vision_pth = "../dataset/19.06_2_cleantest/vision/"
    with open(f"{vision_pth}sample_0.json", "r") as term_vis_file:
        vis = json.load(term_vis_file)
    terminal_coords = []
    term_keys = list(vis["terminals"].keys())
    for t in term_keys:
        terminal_coords.append(vis["terminals"][t]["coordinates"][0])
    
    table_wirepoints, held_wirepoints, _, terminal_points = process_file(vision_pth)
    table_mean_vector, table_covariance_mat = get_max_likelihood(torch.stack(table_wirepoints))
    held_mean_vector, held_covariance_mat = get_max_likelihood(torch.stack(held_wirepoints))
    # term_mean_vector, term_covariance_mat = get_max_likelihood(torch.stack(terminal_points))
    
    # Truncated Multivariate Normal distributions for on_table wires, held wires
    lb = np.array([-1000, -1000, 0.0])
    lb_held = np.array([-1000, -1000, 0.045])
    ub = np.array([1000, 1000, 1000])
    ub_term = np.array([1000, 1000, 0.045])
    trunc_table_dist = TruncatedMVN(table_mean_vector.cpu().numpy(), table_covariance_mat.cpu().numpy(), lb=lb, ub=ub)
    trunc_held_dist = TruncatedMVN(held_mean_vector.cpu().numpy(), held_covariance_mat.cpu().numpy(), lb=lb_held, ub=ub)
    # trunc_term_dist = TruncatedMVN(term_mean_vector.cpu().numpy(), term_covariance_mat.cpu().numpy(), lb=lb, ub=ub_term)
    
    sim_data = SyntheticData(trunc_table_dist, trunc_held_dist, terminal_coords)
    
    sim_data.generate_pick_vision(num_samples=500)
    
if __name__ == "__main__":
    main()
