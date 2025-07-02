import json
import torch
from truncated_normal import TruncatedMVN
from synthetic_dataset import SyntheticData
import numpy as np

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
    # Get reference dataset for distributions
    vision_pth = "../dataset/19.06_2_cleantest/vision/"
    with open(f"{vision_pth}sample_0.json", "r") as term_vis_file:
        vis = json.load(term_vis_file)
    terminal_coords = []
    # Get fixed terminal coordinaes
    for key, value in vis["terminals"].items():
        terminal_coords.append(value["coordinates"])   

    # Process reference dataset, generate max-likelihood for distributions
    table_wirepoints, held_wirepoints, _, terminal_points = process_file(vision_pth)
    table_mean_vector, table_covariance_mat = get_max_likelihood(torch.stack(table_wirepoints))
    held_mean_vector, held_covariance_mat = get_max_likelihood(torch.stack(held_wirepoints))
    
    # Truncated Multivariate Normal distributions for on_table wires, held wires
    lb = np.array([-1000, -1000, 0.0])
    lb_held = np.array([-1000, -1000, 0.045])
    ub = np.array([1000, 1000, 1000])
    ub_term = np.array([1000, 1000, 0.045])
    trunc_table_dist = TruncatedMVN(table_mean_vector.cpu().numpy(), table_covariance_mat.cpu().numpy(), lb=lb, ub=ub)
    trunc_held_dist = TruncatedMVN(held_mean_vector.cpu().numpy(), held_covariance_mat.cpu().numpy(), lb=lb_held, ub=ub)
    
    # Initialize SyntheticData class with distributions
    sim_data = SyntheticData(trunc_table_dist, trunc_held_dist, terminal_coords)

    sim_data.generate_files(num_samples=500, start_samp_num=0, action_type="pick")
    sim_data.generate_files(num_samples=500, start_samp_num=500, action_type="insert")
    sim_data.generate_files(num_samples=500, start_samp_num=1000, action_type="lock")

if __name__ == "__main__":
    main()