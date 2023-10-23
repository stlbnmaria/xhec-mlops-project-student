from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """Given X_train and y_train, uses XGboost to train a model, returning the trained model.

    Parameters
    ----------
    X_train: pd.Dataframe
             Corresponds to the matrix with all the non target columns of the dataset.
    y_train: pd.Series
             Corresponds to the target column of the dataset.

    Returns
    -------
    model : xgb.XGBRregressor
            Trained XGboost model.
    """
    model = xgb.XGBRegressor()

    model.fit(X_train, y_train)

    return model


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
        mse : float
              The Mean Squared Error (RMSE) between the true and predicted values.
        r2 : float
             The R-squared (R2) score, which measures the goodness of fit of the model.
    """
    mse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mse, r2
