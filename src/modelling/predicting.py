import numpy as np
import pandas as pd
import xgboost as xgb
from prefect import task


@task(name="predict", tags=["fails"], retries=3, retry_delay_seconds=60)
def predict(X_data: pd.DataFrame, model: xgb.XGBRegressor) -> np.ndarray:
    """Given X_data and an XGBoost model, returns an array with the values predicted by the model.

    Parameters
    ----------
    X_data: pd.DataFrame
            Input data given to the model
    model:  xgb.XGBRegressor
            Model which will be used to predict the target variable of the input data.

    Returns
    -------
    preds: np.ndarray
           Numpy array containing the predicted values.
    """
    preds = model.predict(X_data)
    return preds
