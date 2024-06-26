import logging

import torch

DEVICE = None


def set_device(device_name):
    if device_name == "auto":
        DEVICE = "auto"
    elif device_name == "cuda":
        assert device_name == "cuda" and torch.cuda.is_available()
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    logging.info(f"Using DEVICE: {DEVICE}")
