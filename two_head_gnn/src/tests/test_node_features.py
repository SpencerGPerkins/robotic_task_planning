import torch
from features.node_features import build_node_features
from features.pos_encoding import SpatialPositionalEncoding

def test_build_node_features_minimal():
    wire_nodes = [{"id": 0, "color": "red", "normalized_coordinates": [0.1, 0.2, 0.3]}]
    terminal_node = {"id": 1, "normalized_coordinates": [0.9, 0.8, 0.7]}
    pos_encoder = SpatialPositionalEncoding(d_model=60, in_dim=3)
    target_info = {"wire_color": "red"}
    label_info = {"global_wire_id": 0}

    wires, terminal, local_id = build_node_features(wire_nodes, terminal_node, pos_encoder, target_info, label_info)
    
    assert isinstance(wires, list)
    assert isinstance(terminal, list)
    assert local_id == 0
