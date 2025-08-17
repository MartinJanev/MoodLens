import os, random, numpy as np, torch

def fix_seed(seed: int = 1337) -> None:
    """
    Fix the random seed for reproducibility across different libraries.
    :param seed: The seed value to set for random number generation.
    :type seed: int
    :return: None

    This function sets the seed for the random module, NumPy, and PyTorch to ensure that
    the results are reproducible across runs. It also sets the environment variable
    PYTHONHASHSEED to the same seed value to ensure consistent hash values across runs.
    This is particularly useful in machine learning experiments where you want to ensure
    that the results are consistent and reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
