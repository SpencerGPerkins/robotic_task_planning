import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from truncated_normal import TruncatedMVN

class MVNStateDistributions:
    
    def __init__(self):
        self.table_distribution = None
        self.held_distribution = None
        
    def process_file(self, vision_pth, num_files=184):
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
            term_keys = list(vision_file["terminals"].keys())
            for key in term_keys:
                terminalpoints.append(torch.tensor(vision_file["terminals"][key]["coordinates"][0]))
        return table_wirepoints, held_wirepoints, inserted_wirepoints, terminalpoints
                
    def MVD(self, in_data:list):
        table_wire_matrix = torch.stack(in_data[0])
        held_wire_matrix = torch.stack(in_data[1])
        self.table_distribution, _, _ = self.get_distribution(table_wire_matrix)
        self.held_distribution, _, _ = self.get_distribution(held_wire_matrix) 
        
    def get_distribution(self, matrix):  
        mean_vector = torch.mean(matrix, dim=0) # Mean for columns (xyz)
        covariance = torch.cov(matrix.T) # Transpose, cov matrix dims should be (3,3)
        # dist = MultivariateNormal(mean_vector, covariance)
        
        return  mean_vector, covariance

def main():
    wire_colors = ["blue", "black", "yellow", "white", "orange", "green", "red"]
    terminals = ["terminal_0", "terminal_1", "terminal_2", "terminal_3", "terminal_4", "terminal_5", "terminal_6", "terminal_7", "terminal_8"]
    
    vision_pth = "../dataset/simulation_data_2/vision/"

    
    
    dist = MVNStateDistributions()
    table_pts, held_pts, inserted_pts, terminal_pts = dist.process_file(vision_pth) # Create distributions for wire states (on_table, held)
    table_mean_vector, table_cov = dist.get_distribution(torch.stack(table_pts))
    held_mean_vector, held_cov = dist.get_distribution(torch.stack(held_pts))
    term_mean_vector, term_cov = dist.get_distribution(torch.stack(terminal_pts))
    # dist.MVD([table_pts, held_pts])
    
    sample_coords_MVN = []
    
    lb = np.array([-1000, -1000, 0.0])
    lb_held = np.array([-1000, -1000, 0.045])
    ub = np.array([1000, 1000, 1000])
    ub_term = np.array([1000, 1000, 0.045])
    trunc_table_dist = TruncatedMVN(table_mean_vector.cpu().numpy(), table_cov.cpu().numpy(), lb=lb, ub=ub)
    trunc_held_dist = TruncatedMVN(held_mean_vector.cpu().numpy(), held_cov.cpu().numpy(), lb=lb_held, ub=ub)
    trunc_term_dist = TruncatedMVN(term_mean_vector.cpu().numpy(), term_cov.cpu().numpy(), lb=lb, ub=ub_term)
    sample_coords_TrunMVN = [trunc_table_dist.sample(100)]
    sample_heldcoords_TrunMVN = [trunc_held_dist.sample(100)]
    sample_termcoords_TrunMVN = [trunc_term_dist.sample(100)]
    # Note: LogMVN doesn't seem to work because shifts scale
    # sample_coords_LogMVN = []
    # for w in range(100):
    #     sample_coords_MVN.append(dist.table_distribution.sample().cpu().numpy().tolist())
        # Note: LogMVN doesn't seem to work because shifts scale
        # sample_wire_LogMVN = torch.exp(dist.table_distribution.sample())
        # sample_coords_LogMVN.append(sample_wire_LogMVN.cpu().numpy().tolist())
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    # for p in range(len(sample_coords_MVN)):
    #     ax.scatter(sample_coords_MVN[p][0], sample_coords_MVN[p][1], sample_coords_MVN[p][2], c="blue")
    for p in range(len(sample_coords_TrunMVN)):
        ax.scatter(sample_coords_TrunMVN[p][0], sample_coords_TrunMVN[p][1], sample_coords_TrunMVN[p][2], c="red")       
    for p in range(len(sample_heldcoords_TrunMVN)):
        ax.scatter(sample_heldcoords_TrunMVN[p][0], sample_heldcoords_TrunMVN[p][1], sample_heldcoords_TrunMVN[p][2], c="green")  
    for p in range(len(sample_termcoords_TrunMVN)):
        ax.scatter(sample_termcoords_TrunMVN[p][0], sample_termcoords_TrunMVN[p][1], sample_termcoords_TrunMVN[p][2], c="blue")  
        
    # # Note: LogMVN doesn't seem to work because shifts scale
    # # for p in range(len(sample_coords_LogMVN)):
    # #     ax.scatter(sample_coords_LogMVN[p][0], sample_coords_LogMVN[p][1], sample_coords_LogMVN[p][2], c="red")
    
    # print(np.mean(np.stack(sample_coords_MVN, axis=0), axis=0))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # for n in range(20):
    #     vision_dict = {
    #         "wires": [],
    #         "terminals": {}
    #     }
    #     num_wires = random.randint(3, 20)
    #     for w in range(num_wires):
    #         color = random.choice(wire_colors)
    #         print(dist.table_distribution)
    #         coords = dist.table_distribution.sample().cpu().numpy().tolist()
    #         print(type(coords))
    #         print(coords)
    #         wire = {
    #             "id": w,
    #             "name": f"{color}_wire",
    #             "color": color,
    #             "state": "on_table",
    #             "coordinates": coords
    #         }
    #         vision_dict["wires"].append(wire)
    #     for t, term_key in enumerate(term_keys):
    #         vision_dict["terminals"][term_key] = {"state": "empty", "coordinates": terminal_coords[t]}
        
    #     with open(f"../dataset/generated_synthetic/vision/sample_{n}.json", "w") as out_file:
    #         json.dump(vision_dict, out_file)

    
if __name__ == "__main__":
    main()




