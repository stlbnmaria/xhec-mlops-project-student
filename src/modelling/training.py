import pandas as pd
import xgboost as xgb
from prefect import task


@task("name=train-model", tags=["fails"], retries=3, retry_delay_seconds=60)
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
