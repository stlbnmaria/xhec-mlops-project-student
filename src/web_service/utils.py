import pickle
from functools import lru_cache
from pathlib import Path

from xgboost import XGBRegressor


@lru_cache
def load_object(filepath: Path) -> XGBRegressor:
    """Given a path, loads the pickle object.

    Parameters
    ----------
    filepath : str
        The path to the object.

    Returns
    -------
    obj : Any
        The de-serialised object.
    """
    with open(filepath, "rb") as in_f:
        obj = pickle.load(in_f)

    return obj
