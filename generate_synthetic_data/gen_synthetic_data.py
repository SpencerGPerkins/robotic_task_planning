from truncated_normal import TruncatedMVN
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
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
            term_keys = list(self.terminals.keys())       
            num_wires = random.randint(3, 20)
            for w in range(num_wires):
                color = random.choice(self.wire_colors)
                coord_list = self.on_table_dist(n=num_wires)
                wire = {
                    "id": w,
                    "name": f"{color}_wire",
                    "color": color,
                    "state": "on_table",
                    "coordinates": coord_list[w]
                }
                vision_dict["wires"].append(wire)
            for t, term_key in enumerate(term_keys):
                vision_dict["terminals"][term_key] = {"state": "empty", "coordinates": terminal_coords[t]}
            
            with open(f"../dataset/generated_synthetic/vision/sample_{n}.json", "w") as out_file:
                json.dump(vision_dict, out_file)
                
def get_max_likelihood(matrix):
    mean_vector = torch.mean(matrix, dim=0) # Mean for columns (xyz)
    covariance_mat = torch.cov(matrix.T) # Transpose, cov matrix dims should be (3,3)
    
    return mean_vector, covariance_mat

def process_file(vision_pth, num_files=434):
    table_wirepoints = []
    held_wirepoints = []
    inserted_wirepoints = []
    terminalpoints = []

    for d in range(num_files):
        with open(f"{vision_pth}/sample_{d}.json", "r") as v_in:
            vision_file = json.load(v_in)
        for wire in vision_file["wires"]: # Get all wire coords
            if wire["state"] == "on_table":
                table_wirepoints.append(torch.tensor(wire["coordinates"][0]))
            elif wire["state"] == "held":
                held_wirepoints.append(torch.tensor(wire["coordinates"][0]))
            elif wire["state"] == "inserted":
                inserted_wirepoints.append(torch.tensor(wire["coordinates"][0]))
            else:
                raise ValueError("Unkown state found in wires...")
            
    return table_wirepoints, held_wirepoints, inserted_wirepoints, terminalpoints



def main():
    wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
    terminals = ["terminal_0", "terminal_1", "terminal_2", "terminal_3", "terminal_4", "terminal_5", "terminal_6", "terminal_7", "terminal_8"]
    
    vision_pth = "../dataset/simulation_data/vision/"
    with open(f"{vision_pth}/sample_0.json", "r") as vision_in:
        vis = json.load(vision_in)
    terminal_coords = []
    term_keys = list(vis["terminals"].keys())
    for t in term_keys:
        terminal_coords.append(vis["terminals"][t]["coordinates"][0])
    
    table_wirepoints, held_wirepoints, _, terminal_points = process_file(vis)
    table_mean_vector, table_covariance_mat = get_max_likelihood(table_wirepoints)
    held_mean_vector, held_covariance_mat = get_max_likelihood(held_wirepoints)
    term_mean_vector, term_covariance_mat = get_max_likelihood(terminal_points)
    
    
    # Truncated Multivariate Normal distributions for on_table wires, held wires
    lb = np.array([-1000, -1000, 0.0])
    lb_held = np.array([-1000, -1000, 0.045])
    ub = np.array([1000, 1000, 1000])
    ub_term = np.arrya([1000, 1000, 0.045])
    trunc_table_dist = TruncatedMVN(table_mean_vector.cpu().numpy(), table_covariance_mat.cpu().numpy(), lb=lb, ub=ub)
    trunc_held_dist = TruncatedMVN(held_mean_vector.cpu().numpy(), held_covariance_mat.cpu().numpy(), lb=lb_held, ub=ub)
    trunc_term_dist = TruncatedMVN(term_mean_vector.cpu().numpy(), term_covariance_mat.cpu().numpy(), lb=lb, ub=)
    
    sim_data = SyntheticData(trunc_table_dist, trunc_held_dist)
    
