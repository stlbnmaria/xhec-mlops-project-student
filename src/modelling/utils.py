import pickle
from typing import Any

def load_pickle(path: str) -> pickle:
    """
    Given a path, loads the pickle object located in that path.

    Parameters
    ----------
    path : string
           String represents the path to the object.

    Returns
    -------
    loaded_obj: pickle object
                Pickle which was contained in the path given as parameter.
    """

    with open(path,"rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any) -> None:
    """
    Given a path and an object, stores the object as a pickle file in the specified path.

    Parameters
    ----------
    path : string
           String represents the path where the pickle object will be stored.

    obj : any
          Represents the object that will be stored.

    Returns
    -------
    None : None
    """

    with open(path, "wb") as f:
        pickle.dump(obj, f)
