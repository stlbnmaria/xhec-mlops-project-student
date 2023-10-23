import pickle
from pathlib import Path

import xgboost as xgb


def load_pickle(path: Path) -> pickle:
    """Given a path, loads the pickle object found in that path.

    Parameters
    ----------
    path : Path
           Represents the path to the object.

    Returns
    -------
    loaded_obj: pickle object
                Pickle which was contained in the path given as parameter.
    """
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: Path, obj: xgb.XGBRegressor) -> None:
    """Given a path and an object, stores the object as a pickle file in the specified path.

    Parameters
    ----------
    path : Path
           Represents the path where the pickle object will be stored.

    obj : LinearRegression
          Represents the linear regression model that will be stored.

    Returns
    -------
    None : None
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
