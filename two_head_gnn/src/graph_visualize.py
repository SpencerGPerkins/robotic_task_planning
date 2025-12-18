import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import os
from graph import TaskGraphHeterogeneous #, GraphHeterogeneous
from tqdm import tqdm
import config


def visualize_graph(graph,target_w_id, action, dataset, sample, save_path):

    G = nx.Graph()

    node_labels = {}
    colors = []
    wire_positions, term_positions = graph.get_positions()
    print(graph.wire_nodes)
    for w, wire in enumerate(graph.wire_nodes):
        G.add_node(w, pos=tuple(wire_positions[w]))
        
        node_labels[w] = f"ID: {wire['id']}\nColor: {wire['color']}\n{wire['state']}"
        if w == target_w_id:
            colors.append("green")
        else:
            colors.append("red")

    t_idx = len(graph.wire_nodes) 
    # G.add_node(t_idx, pos=tuple(graph.terminal_node['normalized_coordinates']))
    # print(f"TERMINAL : {graph.terminal_node['normalized_coordinates']}")
    G.add_node(t_idx, pos=tuple(graph.terminal_node['coordinates']))
    print(f"TERMINAL : {graph.terminal_node['coordinates']}")
    node_labels[t_idx] = f"{graph.terminal_node['name']}\n{graph.terminal_node['state']}"
    colors.append("blue")

    goal_idx = t_idx + 1
    G.add_node(goal_idx, pos=(0,0))
    node_labels[goal_idx] = f"Goal: {action}"
    colors.append("orange")

    edges = graph.get_edge_index().t().tolist()
    edge_labels = {}
    for idx, (src,tgt) in enumerate(edges):
        # feature = graph.edge_attr[idx].item() if graph.edge_attr is not None else 0.0
        feature = graph.edge_attr[idx].numpy().tolist()
        G.add_edge(src, tgt)
        edge_labels[(src, tgt)] = f"{feature}"
        

    pos = nx.kamada_kawai_layout(G)

    for node in G.nodes:
        if node not in pos:
            pos[node] = (0, 0)

    # Draw
    plt.figure(figsize=(20, 17))
    nx.draw(G, pos, node_color=colors, with_labels=True,
            labels=node_labels, node_size=9500, font_size=16, width=3.0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16)

    # Create patches for each color meaning
    legend_patches = [
        mpatches.Patch(color='green', label='Target Wire'),
        mpatches.Patch(color='red',   label='Other Wires'),
        mpatches.Patch(color='blue',  label='Terminal'),
    ]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=12, frameon=True)

    # Save
    save_path = f"{save_path}{dataset}/"
    os.makedirs(save_path, exist_ok=True)
    dir = os.path.join(save_path, f"{sample}.png")
    plt.savefig(dir)
    plt.close()
    # plt.show()

def main(dataset_type: str):

    if dataset_type == "train":
        save_path = f"{config.DATASET_BASE}/graphs/"
        dataset = config.DATASET
        num_samples = config.NUM_SAMPLES
    elif dataset_type == "eval":
        save_path = f"{config.EVAL_DATASET_BASE}/graphs/"
        dataset = config.EVAL_DATASET
        num_samples = config.NUM_EVAL_SAMPLES
    else:
        raise ValueError("Input dataset type must be either 'train' or 'eval'...")
    
    for s in tqdm(range(num_samples)): 
        g_id = s 
        print(f"Graph ID: {s}")
        if dataset_type == "train":
            vision_data = f"{config.VISION_DATA_PATH}sample_{g_id}.json"
            llm_data = f"{config.LLM_DATA_PATH}sample_{g_id}.json"
            label_data = f"{config.LABEL_DATA_PATH}sample_{g_id}.json"
        elif dataset_type == "eval":
            vision_data = f"{config.EVAL_VISION_DATA_PATH}sample_{g_id}.json"
            llm_data = f"{config.EVAL_LLM_DATA_PATH}sample_{g_id}.json"
            label_data = f"{config.EVAL_LABEL_DATA_PATH}sample_{g_id}.json"
        G = TaskGraphHeterogeneous(
            action_primitives = config.ACTION_PRIMS,
            vision_path = vision_data,
            llm_path = llm_data,
            label_path= label_data
        )
        # G = GraphHeterogeneous(
        #     action_primitives=config.ACTION_PRIMS,
        #     vision_path=vision_data,
        #     llm_path=llm_data,
        #     label_path=label_data
        # )
        labels = G.get_labels()
        target_wire_id = labels[1].item()
        action = G.label_info["action"]

        visualize_graph(G, target_wire_id, action, dataset, f"sample_{s}", save_path)

    # g_id = 3012 

    # vision_data = f"{config.VISION_DATA_PATH}sample_{g_id}.json"
    # llm_data = f"{config.LLM_DATA_PATH}sample_{g_id}.json"
    # label_data = f"{config.LABEL_DATA_PATH}sample_{g_id}.json"

    # G = TaskGraphHeterogeneous(
    #     action_primitives = config.ACTION_PRIMS,
    #     vision_path = vision_data,
    #     llm_path = llm_data,
    #     label_path= label_data
    # )
    # labels = G.get_labels()
    # target_wire_id = labels[1].item()
    # action = G.label_info["action"]

    # visualize_graph(G, target_wire_id, action, dataset, f"sample_{g_id}", save_path)

if __name__ == "__main__":
    main("eval")
        




