import logging

import torch


class Device:
    """
    A class to manage device settings for machine learning operations, ensuring the correct
    hardware configuration is used for computations. It supports automatic detection,
    explicit GPU usage if available, or CPU as fallback.

    Attributes:
        DEVICE (torch.device or str): A class-level attribute that stores the currently set device.
    """

    DEVICE = None

    @staticmethod
    def set_device(device_name):
        """
        Sets the device for computation based on the provided device name.

        Args:
            device_name (str): A string indicating the desired device. It can be 'auto' for automatic detection,
                               'cuda' to use GPU (if available), or any other string defaults to CPU.

        This method updates the DEVICE attribute based on the availability and specification of the hardware.
        If 'cuda' is specified, it checks for GPU availability using torch.cuda.is_available() and asserts its presence.
        Logs the device setting for verification and debugging purposes.
        """
        if device_name == "auto":
            Device.DEVICE = "auto"
        elif device_name == "cuda":
            assert (
                device_name == "cuda" and torch.cuda.is_available()
            ), "CUDA is not available on this system."
            Device.DEVICE = torch.device("cuda")
        else:
            Device.DEVICE = torch.device("cpu")

        logging.info(f"Using DEVICE: {Device.DEVICE}")

    @staticmethod
    def get_device():
        """
        Retrieves the currently set device.

        Returns:
            torch.device or str: The device currently set for computation, which could be a torch.device object
                                 for 'cuda' or 'cpu', or 'auto' as a string indicating automatic device selection.
        """
        return Device.DEVICE
