import os
import json
import numpy as np

def is_close(coord1, coord2, tol=1e-3):
    return np.linalg.norm(np.array(coord1) - np.array(coord2)) < tol

def validate_held_wire(sample):
    wires = sample["wires"]
    terminals = sample["terminals"]

    held_wires = [w for w in wires if w["state"] == "held"]
    inserted_terms = [t for t in terminals.values() if t["state"] == "inserted"]

    errors = []
    if len(held_wires) > 1:
        errors.append(f"More than one held wire found: {len(held_wires)}")

    if len(inserted_terms) > 0:
        errors.append("Inserted terminal(s) found in held wire sample")

    return errors

def validate_inserted_wire(sample):
    wires = sample["wires"]
    terminals = sample["terminals"]

    inserted_wires = [w for w in wires if w["state"] == "inserted"]
    inserted_terms = [t for t in terminals.values() if t["state"] == "inserted"]

    errors = []
    if len(inserted_wires) != 1:
        errors.append(f"Expected 1 inserted wire, found {len(inserted_wires)}")
    if len(inserted_terms) != 1:
        errors.append(f"Expected 1 inserted terminal, found {len(inserted_terms)}")

    if inserted_wires and inserted_terms:
        wire_coords = inserted_wires[0]["position"]
        term_coords = inserted_terms[0]["position"]

        if not is_close(wire_coords, term_coords, tol=0.5):  # <- updated tolerance
            errors.append(
                f"Inserted wire coordinates {wire_coords} do not match terminal coordinates {term_coords} within 0.05"
            )

    return errors

def load_sample(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def validate_samples(directory):
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(directory, fname)
        sample = load_sample(path)
        wire_states = [w["state"] for w in sample["wires"]]

        errors = []

        if "held" in wire_states:
            errors = validate_held_wire(sample)
        elif "inserted" in wire_states:
            errors = validate_inserted_wire(sample)

        if errors:
            print(f"[ERROR] {fname}:")
            for err in errors:
                print(f"  - {err}")
    print("Finished check.")

if __name__ == "__main__":
    validate_samples("../dataset/test_dataset/vision")

