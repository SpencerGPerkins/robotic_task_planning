import json
import random
import os
import math

BASE_DIR = "../synthetic_data/4_class"
os.makedirs(f"{BASE_DIR}/vision", exist_ok=True)
os.makedirs(f"{BASE_DIR}/llm", exist_ok=True)
os.makedirs(f"{BASE_DIR}/labels", exist_ok=True)


# Fixed terminal positions
TERMINAL_POSITIONS = {f"terminal_{i}": (random.randint(0, 100), random.randint(0, 100)) for i in range(10)}

WIRE_STATES = ["on_table", "held", "inserted"]
TERMINAL_STATES = ["empty", "inserted", "locked"]
WIRE_COLORS = ["red", "blue", "green", "yellow", "black"]

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def generate_vision_file(num_wires=5):
    vision_data = {"wires": [], "terminals": {}}
    for i in range(num_wires):
        wire_color = random.choice(WIRE_COLORS)
        wire_name = f"{wire_color}_wire"
        state = "on_table"
        coords = (random.randint(0, 100), random.randint(0, 100))
        vision_data["wires"].append({
            "name": wire_name,
            "color": wire_color,
            "state": state,
            "coordinates": coords
        })
    for terminal, coords in TERMINAL_POSITIONS.items():
        vision_data["terminals"][terminal] = {
            "state": "empty",
            "coordinates": coords
        }
    return vision_data

def generate_llm_file(vision_data):
    target_wire = random.choice(vision_data["wires"])
    target_terminal = random.choice(list(vision_data["terminals"].keys()))
    return {"target_wire": target_wire["name"], "target_terminal": target_terminal}

def generate_y_labels(vision_data, llm_data):
    target_wires = [w for w in vision_data["wires"] if w["name"] == llm_data["target_wire"]]
    not_target_wire = [w for w in vision_data["wires"] if w["name"] != llm_data["target_wire"]]
    target_terminal = vision_data["terminals"][llm_data["target_terminal"]]
    not_target_terminal = vision_data["terminals"][llm_data["target_terminal"]]

    if not target_wires:
        raise ValueError(f"No wires found matching target wire type: {llm_data['target_wire']}")

    # If multiple wires exist, choose the closest one to the target terminal
    if len(target_wires) > 1:
        target_wires.sort(key=lambda w: euclidean_distance(w["coordinates"], target_terminal["coordinates"]))
        for i in range(1,len(target_wires)):
            not_target_wire.append(target_wires[i])
            
    # Select the closest wire
    target_wire = target_wires[0]
    target_wire["state"] = random.choice(WIRE_STATES)
    

    # Determine action based on wire state
    if target_wire["state"] == "on_table":
        action = "pick"
        vision_data["terminals"][llm_data["target_terminal"]]["state"] = "empty"
    elif target_wire["state"] == "held":
        action = "insert"
        vision_data["terminals"][llm_data["target_terminal"]]["state"] = "empty"
    elif target_wire["state"] == "inserted":
        target_wire["coordinates"] = vision_data["terminals"][llm_data["target_terminal"]]["coordinates"]
        action = "lock"
        vision_data["terminals"][llm_data["target_terminal"]]["state"] = "inserted"
    else:
        action = "no option"

    return {
        "target_wire": {
            "name": target_wire["name"],
            "color": target_wire["color"],
            "coordinates": target_wire["coordinates"]
        },
        "target_terminal": {
            "name": llm_data["target_terminal"],
            "coordinates": target_terminal["coordinates"]
        },
        "correct_action": action,
        "distance_to_terminal": euclidean_distance(target_wire["coordinates"], target_terminal["coordinates"])
    }

def modify_non_target(vision_data, y_labels):
    available_terminals = [t for t in vision_data["terminals"] if vision_data["terminals"][t]["state"] != "inserted"]
    print(available_terminals)
    print(vision_data["wires"][0]["coordinates"])
    print(y_labels["target_wire"]["coordinates"])
    # non_target_wires = [w for w in vision_data["wires"] if vision_data["wires"][w]["coordinates"] != y_labels["target_wire"]["coordinates"]]
    non_target_wires = []
    for w in range(len(vision_data["wires"])):
        wire_coord = vision_data["wires"][w]["coordinates"]
        
        target_coord = y_labels["target_wire"]["coordinates"]
        if wire_coord[0] != target_coord[0] and wire_coord[1] != target_coord[1]:
            non_target_wires.append(vision_data["wires"][w])
    terminal_threshold = len(available_terminals)
    possible_new_wire_states = ["on_table", "inserted"]
    for w, wire in enumerate(non_target_wires):
        if terminal_threshold > 0:
            wire["state"] = random.choice(possible_new_wire_states)
            if wire["state"] == "inserted":
                print(len(available_terminals))
                index_term = random.randint(0, len(available_terminals)-1)
                print(index_term)
                corres_terminal = random.choice(available_terminals)
                print(corres_terminal)
                available_terminals.pop(index_term)
                wire["coordinates"] = vision_data["terminals"][corres_terminal]["coordinates"]
                vision_data["terminals"][corres_terminal]["state"] = "locked"
                terminal_threshold -= 1
                
        else:
            continue
    
    return vision_data
                
        
    

# Generate 200 samples
for d in range(200):
    vision_data = generate_vision_file()
    llm_data = generate_llm_file(vision_data)
    y_labels = generate_y_labels(vision_data, llm_data)
    vision_data = modify_non_target(vision_data, y_labels)


    # Save files in their respective folders
    with open(f"{BASE_DIR}/vision/sample_{d}.json", "w") as f:
        json.dump(vision_data, f, indent=2)
    with open(f"{BASE_DIR}/llm/sample_{d}.json", "w") as f:
        json.dump(llm_data, f, indent=2)
    with open(f"{BASE_DIR}/labels/sample_{d}.json", "w") as f:
        json.dump(y_labels, f, indent=2)

print("200 synthetic samples generated and sorted into separate directories.")