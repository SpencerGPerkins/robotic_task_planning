# path_config.py
from datetime import datetime

month = datetime.now().month
day = datetime.now().day
year = datetime.now().year

#=========DATASETS-==================
DATASETS_BASE_PATH = "../../dataset/"

TRAINING_DATASETS = {
    "0701_clean": {"path": f"{DATASETS_BASE_PATH}0701_clean/", "num_samples": 276},
    "generated_synthetic_dataset_0709": {"path" :f"{DATASETS_BASE_PATH}generated_synthetic_dataset_0709/", "num_samples": 3000}
}

EVAL_DATASETS = {
    "0701_clean_eval": {"path": f"{DATASETS_BASE_PATH}0701_clean_eval/", "num_samples": 290},
    "generated_eval": {"path": f"{DATASETS_BASE_PATH}generated_eval/", "num_samples": 150}
}

#========Model Parameters============
MODEL_WEIGHTS = {
    "small": f"TwoHeadGAT_small_{month}{day}{year}",
    "medium": f"TwoHeadGAT_medium_{month}{day}{year}"
}

#========Results=====================
RESULTS_BASE = "../docs/results_by_method/"
RESULTS_DIRS = {
    "task_specific": {
        "positional": f"{RESULTS_BASE}task_specific_graph/positional_encoding_based/", "state": f"{RESULTS_BASE}task_specific_graph/state_encoding_based/"
        },
    "task_agnostic": {
        "positional": f"{RESULTS_BASE}task_agnostic_graph/positional_encoding_based/",
        "state": f"{RESULTS_BASE}task_agnostic_graph/state_encoding_based/"
    }
}

