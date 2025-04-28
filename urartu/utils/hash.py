import torch  # NOQA
from typing import Dict
import hashlib


def dict_to_8char_hash(d: Dict):
    """
    Generates an 8-character alphanumeric hash string from a dictionary by converting it
    to a string and hashing it with SHA-256.

    Args:
        d (Dict): The dictionary to hash.

    Returns:
        str: An 8-character alphanumeric string representation of the hash.

    Example:
        >>> dict_to_8char_hash({'model': 'gpt2', 'layers': 12})
        '3f8a2c7b'
    """
    dict_str = str(sorted(d.items()))
    hash_obj = hashlib.sha256(dict_str.encode())
    hash_hex = hash_obj.hexdigest()
    
    return hash_hex[:8]
