from graph.graph_constructor import TaskGraphHeterogeneous
import torch

def test_graph_build():
    g = TaskGraphHeterogeneous(
        action_primitives=["pick", "insert", "lock", "putdown"],
        vision_path="tests/data/vision.json",
        llm_path="tests/data/llm.json",
        label_path="tests/data/labels.json"
    )

    X_w, X_t = g.get_node_features()
    assert X_w.shape[1] > 2
    assert isinstance(g.get_edge_index(), torch.Tensor)
    assert g.get_labels()[0].item() >= 0
