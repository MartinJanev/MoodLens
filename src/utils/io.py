import os, torch
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    """
    Ensure that the directory exists, creating it if necessary.
    If the directory already exists, this function does nothing.
    :param path: The directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """
    Save the model state to a checkpoint file.
    :param path: The file path where the checkpoint will be saved.
    :param state: The state dictionary containing model parameters and other information.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a model state from a checkpoint file.
    :param path: The file path from which the checkpoint will be loaded.
    :return: The state dictionary containing model parameters and other information.
    """
    return torch.load(path, map_location="cpu")
