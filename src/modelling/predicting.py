import numpy as np
import pandas as pd
import xgboost as xgb
from preprocessing import transform_data
from training import predict


def predict_pipeline(input_data: pd.DataFrame, model: xgb.XGBRegressor) -> np.ndarray:
    """Pipeline to predict on the input data using the given model.

    Parameters
    -------
    input_data : pd.Dataframe
                 Pandas input dataframe.

    model : xgb.XGBRegressor
            Model used to predict on the input data.

    Returns
    -------
    y : np.ndarray
        Array of predicted target values.
    """
    df = transform_data(input_data)
    y = predict(df, model)
    return y
