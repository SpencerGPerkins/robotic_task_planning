# config.py
import torch

#=====General config==================================================#
# Action Primitives
ACTION_PRIMS = ["pick", "insert", "lock", "putdown"]

# Model params
MODEL_SIZE = "small"
HIDDEN_DIM = 64
NUM_ACTIONS = 4

# Checkpoint
CHECKPOINT_PATH = f"TwoHeadGAT_0704_{MODEL_SIZE}_0701_clean_eval.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=====Training / validation config====================================#
# Data paths
DATASET = "0701_clean_eval"
VISION_DATA_PATH = f"../../dataset/{DATASET}/vision/"
LLM_DATA_PATH = f"../../dataset/{DATASET}/llm/"
LABEL_DATA_PATH = f"../../dataset/{DATASET}/labels/"


# Training params
NUM_SAMPLES = 290
BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Save paths
SAVE_MODEL_WEIGHTS = f"TwoHeadGAT_0704_{MODEL_SIZE}_{DATASET}.pth"
SAVE_RESULTS_HEAD = f"../docs/TwoHead_training_results/TwoHeadGAT_{MODEL_SIZE}"


#======Testing config=================================================# 
EVAL_DATASET = "generated_eval"
EVAL_VISION_DATA_PATH = f"../../dataset/{EVAL_DATASET}/vision/"
EVAL_LLM_DATA_PATH = f"../../dataset/{EVAL_DATASET}/llm/"
EVAL_LABEL_DATA_PATH = f"../../dataset/{EVAL_DATASET}/labels/"
NUM_EVAL_SAMPLES = 150
SAVE_EVAL_RESULTS_HEAD = f"../docs/TwoHead_Evaluation_results/TwoHeadGAT_{MODEL_SIZE}"

