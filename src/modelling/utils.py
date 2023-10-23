import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import xgboost as xgb
from prefect import task
from sklearn.metrics import mean_squared_error, r2_score


@task(name="load-pickle", tags=["fails"], retries=3, retry_delay_seconds=60)
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


@task(name="save-pickle", tags=["fails"], retries=3, retry_delay_seconds=60)
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


@task(name="evaluate-model", tags=["fails"], retries=3, retry_delay_seconds=60)
def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """Evaluate the model by calculating Mean Squared Error (MSE) and R-squared (R2) scores.

    Parameters
    -----------
        y_true : pd.Series
                 The true target values.
        y_pred : pd.Series
                 The predicted target values.

    Returns
    --------
        rmse : float
              The Root Mean Squared Error (RMSE) between the true and predicted values.
        r2 : float
             The R-squared (R2) score, which measures the goodness of fit of the model.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2
