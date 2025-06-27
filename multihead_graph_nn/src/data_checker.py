import json
import os
from collections import defaultdict

# Paths
v_pth = "../synthetic_data/4_class/vision/"
llm_pth = "../synthetic_data/4_class/llm/"
l_pth = "../synthetic_data/4_class/labels/"

# Action-to-wire-state mapping
ACTION_STATE_RULES = {
    "pick": "on_table",
    "insert": "held",
    "lock": "inserted",
    "putdown": "on_table"
}

def check_identical_wire_coords(sample, sample_index, issues_dict):
    seen_coords = set()
    for wire in sample.get("wires", []):
        coord = tuple(wire["coordinates"])
        if coord in seen_coords:
            issues_dict["identical wire coords"].append(sample_index)
            return
        seen_coords.add(coord)

def check_terminal_wire_state_match(sample, sample_index, issues_dict):
    terminals = {
        tuple(terminal["coordinates"]): terminal["state"]
        for terminal in sample.get("terminals", {}).values()
    }

    for wire in sample.get("wires", []):
        coord = tuple(wire["coordinates"])
        wire_state = wire["state"]
        terminal_state = terminals.get(coord)

        if wire_state == "inserted":
            if terminal_state not in {"inserted", "locked"}:
                issues_dict["mismatch terminal_wire state"].append(sample_index)
                return


def check_label_action_vs_wire_state(vision_sample, label_sample, sample_index, issues_dict):
    target_coord = tuple(label_sample["target_wire"]["coordinates"])
    expected_state = ACTION_STATE_RULES.get(label_sample["correct_action"])

    # Find wire in vision with matching coords
    for wire in vision_sample.get("wires", []):
        if tuple(wire["coordinates"]) == target_coord:
            if wire["state"] != expected_state:
                issues_dict["label-action wire-state mismatch"].append(sample_index)
            return

    # If no matching wire found
    issues_dict["label-action wire-state mismatch"].append(sample_index)

def process_samples(vision_dir, label_dir, num_samples=200):
    issues = defaultdict(list)
    for idx in range(num_samples):
        vision_path = os.path.join(vision_dir, f"sample_{idx}.json")
        label_path = os.path.join(label_dir, f"sample_{idx}.json")

        if not os.path.isfile(vision_path) or not os.path.isfile(label_path):
            print(f"Missing file at index {idx}")
            continue

        with open(vision_path, "r") as vf, open(label_path, "r") as lf:
            vision_sample = json.load(vf)
            label_sample = json.load(lf)

        check_identical_wire_coords(vision_sample, idx, issues)
        check_terminal_wire_state_match(vision_sample, idx, issues)
        check_label_action_vs_wire_state(vision_sample, label_sample, idx, issues)
        
    return dict(issues)

# Run the script
if __name__ == "__main__":
    issues_found = process_samples(v_pth, l_pth, num_samples=200)
    print(json.dumps(issues_found, indent=2))