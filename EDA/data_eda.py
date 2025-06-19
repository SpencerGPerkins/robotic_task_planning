
"""Created 6/18/2025
EDA for data collected via Isaac Sim (using deterministic action prediction model)

Simulation = full task (e.g. goal=lock, simulation=[pick,insert,lock])

## Script Details: ##

SimHist stores simulation details, all xyz coords collected from data directories (vision, labels, end_sim_state)
        Computes mean vectors and covariance matrices from labels file for analysis

Script produces the following saved files:
xyz distribution plots for vision, labels, and end_sim_state (optional show and save)

json file for EDA of labels
json files for wire and terminal coordinates

"""
import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class SimHist:
    def __init__(self):
        self.simulation_length = [] # Number of iters for full simulation
        self.final_iter_idx = []
        
        # History for wire points, keys are filenames
        self.wire_points = {
            "vision": {"table_wirepoints": [], "held_wirepoints": [], "inserted_wirepoints": []},
            "labels": {"table_wirepoints": [], "held_wirepoints": [], "inserted_wirepoints": []},
            "end_sim_state": {"end_wirepoints": []}
            }
        self.terminal_points = {
            "vision": {"terminalpoints": []},
            "labels": {"terminalpoints": []},
            "end_sim_state": {"end_terminalpoints": []}
        }
            
        # Distribution of action labels
        self.label_action_counts = {"pick":0, "insert": 0, "lock": 0, "putdown": 0}
        
        # State-based stats for wire xyz (based on label file)
        self.label_wire_meanvectors = {"on_table":[], "held": [], "inserted": []}
        self.label_wire_covmat = {"on_table": [], "held": [], "inserted": []}
        
    def process_file(self, pth, file_name:str, num_files):

        if file_name == "vision":
            sim_len_counter = 0
            previous_sim = 0
            wire_key = "wires"
            terminal_key = "terminals"
            for d in range(num_files):
                with open(f"{pth}{file_name}/sample_{d}.json", "r") as v_in:
                    file = json.load(v_in)
                current_sim = file["metadata"]["simulation_id"]
                if current_sim != previous_sim:
                    self.simulation_length.append(sim_len_counter)
                    self.final_iter_idx.append(d)
                    previous_sim = current_sim
                    sim_len_counter = 0
                else:
                    sim_len_counter += 1

                for wire in file[wire_key]: # Get all wire coords
                    if wire["state"] == "on_table":
                        self.wire_points[file_name]["table_wirepoints"].append(wire["coordinates"])
                    elif wire["state"] == "held":
                       self. wire_points[file_name]["held_wirepoints"].append(wire["coordinates"])
                    elif wire["state"] == "inserted":
                        self.wire_points[file_name]["inserted_wirepoints"].append(wire["coordinates"])
                    else:
                        raise ValueError("Unkown state found in wires...")
                term_keys = list(file[terminal_key].keys())
                for key in term_keys:
                    self.terminal_points[file_name]["terminalpoints"].append(file["terminals"][key]["coordinates"])
                    
            return self.wire_points[file_name], self.terminal_points[file_name]
        
        elif file_name == "labels":
            wire_key = "target_wire"
            terminal_key = "target_terminal"
            
            # Dictionary for tensors to compute mean vectors and covariance matrices
            label_dict = {
                "wires": {"table_wirepoints": [], "held_wirepoints": [], "inserted_wirepoints": []},
                "terminals" :{"terminalpoints": []}  
            }
            for d in range(num_files):
                with open(f"{pth}{file_name}/sample_{d}.json", "r") as v_in:
                    file = json.load(v_in)
                if file["correct_action"] == "pick":
                    self.label_action_counts[file["correct_action"]] += 1
                    self.wire_points[file_name]["table_wirepoints"].append(file[wire_key]["coordinates"])
                    label_dict["wires"]["table_wirepoints"].append(torch.tensor(file[wire_key]["coordinates"]))
                elif file["correct_action"] == "insert" or file["correct_action"] == "putdown":
                    self.label_action_counts[file["correct_action"]] += 1
                    self.wire_points[file_name]["held_wirepoints"].append(file[wire_key]["coordinates"])
                    label_dict["wires"]["held_wirepoints"].append(torch.tensor(file[wire_key]["coordinates"]))
                elif file["correct_action"] == "lock":
                    self.label_action_counts[file["correct_action"]] += 1
                    self.wire_points[file_name]["inserted_wirepoints"].append(file[wire_key]["coordinates"])
                    label_dict["wires"]["inserted_wirepoints"].append(torch.tensor(file[wire_key]["coordinates"]))
                else:
                    raise ValueError("Unkown state found in wires...")

                self.terminal_points[file_name]["terminalpoints"].append(file[terminal_key]["coordinates"])
                label_dict["terminals"]["terminalpoints"].append(file[terminal_key]["coordinates"])
                
            # State based Mean Vectors xyz coordinates 
            self.label_wire_meanvectors["on_table"] = torch.mean(torch.stack(label_dict["wires"]["table_wirepoints"])).cpu().numpy().tolist()
            self.label_wire_meanvectors["held"] = torch.mean(torch.stack(label_dict["wires"]["held_wirepoints"])).cpu().numpy().tolist()
            self.label_wire_meanvectors["inserted"] = torch.mean(torch.stack(label_dict["wires"]["inserted_wirepoints"])).cpu().numpy().tolist() 
            
            # State based Covariance Matrix
            self.label_wire_covmat["on_table"] = torch.cov(torch.stack(label_dict["wires"]["table_wirepoints"]).T).cpu().numpy().tolist()
            self.label_wire_covmat["held"] = torch.cov(torch.stack(label_dict["wires"]["held_wirepoints"]).T).cpu().numpy().tolist()
            self.label_wire_covmat["inserted"] = torch.cov(torch.stack(label_dict["wires"]["inserted_wirepoints"]).T).cpu().numpy().tolist() 
                 
            return self.wire_points[file_name], self.terminal_points[file_name]
        
        elif file_name == "end_sim_state":
            wire_key = "wire"
            terminal_key = "terminal"
            for d in range(num_files):
                with open(f"{pth}{file_name}/simulation_cycle_{d}.json", "r") as v_in:
                    file = json.load(v_in)
                self.wire_points[file_name]["end_wirepoints"].append(file["wire"]["position"])
                self.terminal_points[file_name]["end_terminalpoints"].append(file["terminal"]["position"])
                
            return self.wire_points[file_name]["end_wirepoints"], self.terminal_points[file_name]["end_terminalpoints"]

def vision_labels_eda(table_pts, held_pts, inserted_pts, terminal_pts, save_filename, save_ending, show_plot=False): 
    """ Plots for vision and label files"""     
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    
    for p in range(len(table_pts)):
        ax.scatter(table_pts[p][0], table_pts[p][1], table_pts[p][2], c="red")       
    for p in range(len(held_pts)):
        ax.scatter(held_pts[p][0], held_pts[p][1], held_pts[p][2], c="green")  
    for p in range(len(inserted_pts)):
        ax.scatter(inserted_pts[p][0], inserted_pts[p][1], inserted_pts[p][2], c="yellow")  
    for p in range(len(terminal_pts)):
        ax.scatter(terminal_pts[p][0], terminal_pts[p][1], terminal_pts[p][2], c="blue")  

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(f"eda/figs/{save_filename}_eda_{save_ending}.png")
    print(f"Plot saved at : eda/figs/{save_filename}_eda_{save_ending}.png")
    
    if show_plot:
        plt.show()
        
def end_state_eda(wire, term, save_ending, show_plot=False):
    """Plots for End State files"""
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    
    for p in range(len(wire)):
        ax.scatter(wire[p][0], wire[p][1], wire[p][2], c="red")       
    for p in range(len(term)):
        ax.scatter(term[p][0], term[p][1], term[p][2], c="blue")  

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(f"eda/figs/sim_end_state_eda_{save_ending}.png")
    print(f"File saved at : eda/figs/sim_end_state_eda_{save_ending}.png")
    
    if show_plot:
        plt.show()    
            
def main(plot=True):
    now = datetime.now()
    month, day, hour, minute = now.month, now.day, now.hour, now.minute
    
    save_title_ending = f"{month}_{day}_{hour}{minute}"
    hist = SimHist()
    file_names = ["vision", "labels"]
    # file_names = ["vision"]
    pth = "../dataset/19.06_2_cleantest/"
    
    
    # Process Vision and labels files
    for filename in file_names:
        vl_directory = pth + filename +"/"
        num_files_vl = len([f for f in os.listdir(vl_directory) if os.path.isfile(os.path.join(vl_directory, f))])
        wire_pts, terminal_pts = hist.process_file(pth, filename, num_files=num_files_vl) # Create distributions for wire states (on_table, held)
        table_wires = wire_pts["table_wirepoints"]

        held_wires = wire_pts["held_wirepoints"]
        inserted_wires = wire_pts["inserted_wirepoints"]
        terminals = terminal_pts["terminalpoints"]
        print(len(table_wires), len(held_wires), len(inserted_wires), len(terminals))
        
        if plot:
            vision_labels_eda(table_wires, held_wires, inserted_wires, terminals, save_filename=filename, save_ending=save_title_ending, show_plot=True)        
    print(f"Total Simulations: {len(hist.simulation_length)}")
    
    # Process end_sim_state file
    cycle_directory = pth + "end_sim_state/"
    num_files_cycle = len([f for f in os.listdir(cycle_directory) if os.path.isfile(os.path.join(cycle_directory, f))])
    end_wirepts, end_termpts = hist.process_file(pth,"end_sim_state", num_files=num_files_cycle)
    if plot:
        end_state_eda(end_wirepts, end_termpts, save_ending=save_title_ending, show_plot=True)
    
    # Dict for writing Label data
    labels_eda_dict = {
        "action_totals":hist.label_action_counts,
        "xyz_means":hist.label_wire_meanvectors,
        "wire_cov_matrices": hist.label_wire_covmat
    }
    
    # Save action totals, xyz mean vectors for wires, xyz covariance matrices for wires 
    # (hist.label_action_counts, hist.label_wire_meanvectors, hist.label_wire_covmat)
    with open(f"eda/meta/labels_eda_{save_title_ending}.json", "w") as write_file:
        json.dump(labels_eda_dict, write_file)
    # Save wire points (hist.wire_points)
    with open(f"eda/meta/wirepoints_{save_title_ending}.json", "w") as write_wires:
        json.dump(hist.wire_points, write_wires)
    # Save terminal points (hist.terminal_points)
    with open(f"eda/meta/terminalpoints_{save_title_ending}.json", "w") as write_terminals:
        json.dump(hist.terminal_points, write_terminals)
    
    print("DATA Processed, Files Saved.")
        
if __name__ == "__main__":
    main()
