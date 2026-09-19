import random

import numpy as np
import torch

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    OUTPUT_DIR
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def validate_project_paths():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: "
            f"{TRAIN_DIR.resolve()}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: "
            f"{TEST_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def clear_device_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()
