import numpy as np
import pandas as pd

from config.config import MODEL_PATH
from src.web_service.lib.models import ModelInput, ModelOutput
from src.web_service.utils import load_object


def get_input_df(payload: ModelInput) -> pd.DataFrame:
    """Convert 'ModelInput' object to the pandas dataframe.

    Parameters
    ----------
    payload : ModelInput
        The model input.

    Returns
    -------
    input_df : pd.DataFrame
        The model input converted to the pandas DataFrame.
    """
    input_df = pd.DataFrame(
        [
            {
                "Length": payload.length,
                "Diameter": payload.diameter,
                "Height": payload.height,
                "Whole weight": payload.whole_weight,
                "Shucked weight": payload.shucked_weight,
                "Viscera weight": payload.viscera_weight,
                "Shell weight": payload.shell_weight,
                "Sex": payload.sex,
            }
        ]
    )

    return input_df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to be preprocessed.

    Returns
    -------
    df : pd.DataFrame
        Processed dataframe.
    """
    df["Sex_I"] = np.where(df["Sex"] == "I", 1, 0)
    df["Sex_M"] = np.where(df["Sex"] == "M", 1, 0)
    df["Sex_F"] = np.where(df["Sex"] == "F", 1, 0)
    df = df.drop(columns="Sex")
    return df


def infer_age(payload: ModelInput) -> ModelOutput:
    """Predict abalone age.

    Parameters
    ----------
    payload : ModelInput
        Model input object. Encapsulates input features.

    Returns
    -------
    prediction : ModelOutput
        Model output object. Encapsulates predicted age.
    """
    model = load_object(MODEL_PATH)

    df = get_input_df(payload)
    x = preprocessing(df)

    y_pred = model.predict(x)[0]
    prediction = ModelOutput(abalone_age=y_pred)

    return prediction
