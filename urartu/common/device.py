import logging

import torch


class Device:
    DEVICE = None

    @staticmethod
    def set_device(device_name):
        if device_name == "auto":
            Device.DEVICE = "auto"
        elif device_name == "cuda":
            assert device_name == "cuda" and torch.cuda.is_available()
            Device.DEVICE = torch.device("cuda")
        else:
            Device.DEVICE = torch.device("cpu")

        logging.info(f"Using DEVICE: {Device.DEVICE}")

    @staticmethod
    def get_device():
        return Device.DEVICE
