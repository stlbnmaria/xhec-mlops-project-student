import numpy as np
import pandas as pd
import xgboost as xgb
from preprocessing import extract_x_y_split, transform_data


def predict_pipeline(input_data: pd.DataFrame, model: xgb.XGBRegressor) -> np.ndarray:
    """Pipeline to predict on the input data using the given model.

    Parameters
    -------
    input_data : pd.Dataframe
                 Pandas input dataframe.

    model : xgb.XGBRegressor
            Model used to predict on the input data.

    Returns:
    -------
    y : np.ndarray
        Array of predicted target values.
    """
    df = transform_data(input_data)
    X_train, X_test, y_train, y_test = extract_x_y_split(df)
    y = model.predict(X_test)
    return y
