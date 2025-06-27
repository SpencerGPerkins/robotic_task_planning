import json
import os

# Define paths to the three directories
labels_dir = "../synthetic_data/4_class/labels"
llm_dir = "../synthetic_data/4_class/llm"
vision_dir = "../synthetic_data/4_class/vision"

# Function to load JSON files
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
action_counters = {"pick": 0, "insert": 0, "lock": 0, "putdown":0}
lock_counter = 0
insert_counter = 0
pick_counter = 0

# Iterate through all data points
for f, filename in enumerate(os.listdir(labels_dir)):

    if filename.endswith(".json"):  # Ensure we are reading JSON files
        label_path = os.path.join(labels_dir, filename)
        llm_path = os.path.join(llm_dir, filename)
        vision_path = os.path.join(vision_dir, filename)

        # Load JSON data
        labels = load_json(label_path)
        llm = load_json(llm_path)
        vision = load_json(vision_path)
        action_counters[labels["correct_action"]] += 1

        # Consistency check between files
        assert labels["target_wire"]["name"] == llm["target_wire"], f"Mismatch in target wire for {filename}"
        assert labels["target_terminal"]["name"] == llm["target_terminal"], f"Mismatch in target terminal for {filename}"

        # Check vision conditions
        terminal_keys = vision["terminals"].keys()
        inserted_wires = [w for w in vision["wires"] if w["state"] == "inserted"]
        inserted_terminals = [key for key in terminal_keys if vision["terminals"][key]["state"] == "inserted"]
        locked_terminals = [key for key in terminal_keys if vision["terminals"][key]["state"] == "locked"]
        print(len(inserted_wires))
        print(len(locked_terminals))
        print(len(inserted_wires) - len(locked_terminals))
        assert len(inserted_wires) - len(locked_terminals) <= 1, f"Check non-target inserted wires in {filename}"
        assert len(inserted_terminals) <= 1, f"More than one inserted wire in {filename}"
        
        for wire in vision["wires"]:
            if wire["coordinates"] == labels["target_wire"]["coordinates"]:
                continue
            elif wire["state"] in ["inserted", "locked"] and wire["coordinates"] != labels["target_wire"]["coordinates"]:
                wire_coords = tuple(wire["coordinates"])
                terminal_match = any(
                    tuple(terminal["coordinates"]) == wire_coords and terminal["state"] == "locked"
                    for terminal in vision["terminals"].values()
                )
                assert terminal_match, f"Wire at {wire_coords} does not match any locked terminal in {filename}"

        # Check label conditions
        if labels["correct_action"] == "lock":
            assert labels["distance_to_terminal"] == 0.0, f"Distance should be 0.0 for lock action in {filename}"
            print(labels["target_wire"]["coordinates"], labels["target_terminal"]["coordinates"])
            assert labels["target_wire"]["coordinates"] == labels["target_terminal"]["coordinates"], \
                f"Target wire and terminal coordinates should match for lock action in {filename}"
    print(f"{f+1} file(s) checked...")
    print(f"Totals: Pick {action_counters['pick']}, Insert {action_counters['insert']}, Lock {action_counters['lock']}, Putdown {action_counters['putdown']}")
print("All consistency checks passed!")