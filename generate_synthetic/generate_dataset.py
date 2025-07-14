import json
import torch
import numpy as np
from truncated_normal import TruncatedMVN
from synthetic_dataset import SyntheticData
from config import DATASET_CONFIG

def get_max_likelihood(matrix):
    mean_vector = torch.mean(matrix, dim=0)
    covariance_mat = torch.cov(matrix.T)
    return mean_vector, covariance_mat

def process_reference_files(vision_pth, num_files):
    table_wirepoints, held_wirepoints, inserted_wirepoints, terminalpoints = [], [], [], []

    for i in range(num_files):
        with open(f"{vision_pth}/sample_{i}.json", "r") as v_in:
            vision_data = json.load(v_in)
        for wire in vision_data["wires"]:
            pos = torch.tensor(wire["position"])
            if wire["state"] == "on_table":
                table_wirepoints.append(pos)
            elif wire["state"] == "held":
                held_wirepoints.append(pos)
            elif wire["state"] == "inserted":
                inserted_wirepoints.append(pos)
        for term in vision_data["terminals"].values():
            terminalpoints.append(torch.tensor(term["position"]))
            
    return table_wirepoints, held_wirepoints, inserted_wirepoints, terminalpoints

def main(config):
    vis_path = config["input_vision_path"]
    num_ref = config["num_reference_files"]
    
    with open(f"{vis_path}/sample_0.json") as f:
        term_coords = [term["position"] for term in json.load(f)["terminals"].values()]
    
    table_pts, held_pts, inserted_pts, _ = process_reference_files(vis_path, num_ref)
    table_mv, table_cov = get_max_likelihood(torch.stack(table_pts))
    held_mv, held_cov = get_max_likelihood(torch.stack(held_pts))
    inserted_mv, inserted_cov = get_max_likelihood(torch.stack(inserted_pts))

    lb = np.array([-1000, -1000, 0.0])
    ub = np.array([1000, 1000, 1000])
    lb_held = np.array([-1000, -1000, 0.045])
    ub_term = np.array([1000, 1000, 0.045])

    trunc_table = TruncatedMVN(table_mv.numpy(), table_cov.numpy(), lb=lb, ub=ub)
    trunc_held = TruncatedMVN(held_mv.numpy(), held_cov.numpy(), lb=lb_held, ub=ub)
    trunc_inserted = TruncatedMVN(inserted_mv.numpy(), inserted_cov.numpy(), lb=lb, ub=ub_term)

    sim_data = SyntheticData(trunc_table, trunc_held, trunc_inserted, term_coords, config)

    start = config["start_samp_number"]
    for action in config["actions"]:
        sim_data.generate_files(
            num_samples=config["num_samples_per_action"],
            start_samp_num=start,
            action_type=action
        )
        start += config["num_samples_per_action"]

        sim_data.generate_files_onepercolor(
            num_samples=config["num_samples_per_action"],
            start_samp_num=start,
            action_type=action
        )
        start += config["num_samples_per_action"]

if __name__ == "__main__":
    from config import DATASET_CONFIG
    main(DATASET_CONFIG)
