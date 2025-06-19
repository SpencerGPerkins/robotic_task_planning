import json
import os

def process_coords(vis, lab, sample_number):
    term_coords = lab["target_terminal"]["coordinates"][0]
    wire_id = lab["target_wire"]["ID"]
    if lab["target_wire"]["coordinates"][0] != term_coords:
        lab["target_wire"]["coordinates"][0] = term_coords
        print(f"Sample {sample_number} label file changed")
    else:
        print(f"terminal coords ({term_coords}) = wire coords({lab['target_wire']['coordinates'][0]}) in label")
        
    if vis["wires"][wire_id]["coordinates"][0] != term_coords:
        vis["wires"][wire_id]["coordinates"][0] = term_coords
        print(f"Sample {sample_number} vision file changed")
    else:
        print(f"terminal coords ({term_coords}) = wire coords({vis['wires'][wire_id]['coordinates'][0]}) in vision")
    
    with open(f"/home/spencer/Documents/research/hucenrotia_lab/working_directory/task_plan/working_dir/dataset/synthetic_data/vision/sample_{s}.json", "w") as mod_vis:
        json.dump(vis, mod_vis)
    with open(f"/home/spencer/Documents/research/hucenrotia_lab/working_directory/task_plan/working_dir/dataset/synthetic_data/labels/sample_{s}.json", "w") as mod_lab:
        json.dump(lab, mod_lab)    

global_path = "/home/spencer/Documents/research/hucenrotia_lab/working_directory/task_plan/working_dir/dataset/synthetic_data/"
# Get current dataset length
lst = os.listdir(f"{global_path}vision") 
num_samples = len(lst)

for s in range(num_samples):
    with open(f"{global_path}vision/sample_{s}.json", "r") as vision_in:
        vision_data = json.load(vision_in)
    with open(f"{global_path}llm/sample_{s}.json", "r") as llm_in:
        llm_data= json.load(llm_in)
    with open(f"{global_path}labels/sample_{s}.json", "r") as labels_in:
        label_data = json.load(labels_in)
    
    action = label_data["correct_action"]
    
    if action != "lock":
        continue
    elif action == "lock":
        process_coords(vision_data, label_data, s)
    else:
        print(f"Uknown action in file {s}")
    