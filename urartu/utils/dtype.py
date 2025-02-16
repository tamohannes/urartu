import torch  # NOQA


def eval_dtype(string: str):
    """
    Evaluates a string representation of a PyTorch data type to return the corresponding
    data type object. This function is primarily used to convert configuration strings
    into actual PyTorch data types.

    Args:
        string (str): A string that represents a PyTorch data type (e.g., 'torch.float32').

    Returns:
        torch.dtype: The PyTorch data type object corresponding to the input string.

    Raises:
        SyntaxError: If the string does not represent a valid PyTorch data type.
        NameError: If the string refers to a data type or object not defined in the scope.

    Example:
        >>> eval_dtype('torch.float32')
        torch.float32
    """
    return eval(string)
