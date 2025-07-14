# config.py
import torch
import path_config

#=====General config==================================================#
# Action Primitives
ACTION_PRIMS = ["pick", "insert", "lock", "putdown"]

# Model params
MODEL_SIZE = "small"
HIDDEN_DIM = 64
NUM_ACTIONS = 4

GRAPH_TYPE = "task_specific" # Supports "task_specific", "task_agnostic"
NODE_FEATURE_TYPE = "positional" # Supports "positional", "state"

# Checkpoint
CHECKPOINT_PATH = f"{path_config.MODEL_WEIGHTS[MODEL_SIZE]}_{NODE_FEATURE_TYPE}_{GRAPH_TYPE}.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#=====Training / validation config====================================#
# Data paths
DATASET = "generated_synthetic_dataset_0709"
DATASET_BASE = path_config.TRAINING_DATASETS[DATASET]["path"]
VISION_DATA_PATH = f"{DATASET_BASE}/vision/"
LLM_DATA_PATH = f"{DATASET_BASE}/llm/"
LABEL_DATA_PATH = f"{DATASET_BASE}/labels/"


# Training params
NUM_SAMPLES = path_config.TRAINING_DATASETS[DATASET]["num_samples"]
BATCH_SIZE = 1
NUM_EPOCHS = 25
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Save paths
SAVE_MODEL_WEIGHTS = CHECKPOINT_PATH
SAVE_RESULTS_HEAD = f"{path_config.RESULTS_DIRS[GRAPH_TYPE][NODE_FEATURE_TYPE]}{MODEL_SIZE}_model/training_results/"


#======Testing config=================================================# 
EVAL_DATASET = "0701_clean_eval"
EVAL_DATASET_BASE = path_config.EVAL_DATASETS[EVAL_DATASET]["path"]

EVAL_VISION_DATA_PATH = f"{EVAL_DATASET_BASE}/vision/"
EVAL_LLM_DATA_PATH = f"{EVAL_DATASET_BASE}/llm/"
EVAL_LABEL_DATA_PATH = f"{EVAL_DATASET_BASE}/labels/"

NUM_EVAL_SAMPLES = path_config.EVAL_DATASETS[EVAL_DATASET]["num_samples"]

SAVE_EVAL_RESULTS_HEAD = f"{path_config.RESULTS_DIRS[GRAPH_TYPE][NODE_FEATURE_TYPE]}{MODEL_SIZE}_model/evaluation_results/"

