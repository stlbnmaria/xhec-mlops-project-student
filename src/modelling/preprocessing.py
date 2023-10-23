from pathlib import Path
from typing import Tuple

import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split


@task(name="read-data", tags=["fails"], retries=3, retry_delay_seconds=60)
def read_data(path: Path) -> pd.DataFrame:
    """Given a path, loads the data as a pandas dataframe.

    Parameters
    ----------
    path : Path
           Represents the path to the csv file.

    Returns
    -------
    df : pd.Dataframe
         Pandas dataframe.
    """
    # read the csv
    df = pd.read_csv(path)
    return df


@task(name="transform-data", tags=["fails"], retries=3, retry_delay_seconds=60)
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe, applies transformations and returns the transformed dataframe.

    Parameters
    -------
    df : pd.Dataframe
         Pandas input dataframe.

    Returns
    -------
    df : pd.Dataframe
         Transformed dataframe.
    """
    # add age column
    df["age"] = df["Rings"] + 1.5
    # one-hot-encoding sex column
    df = pd.get_dummies(df, columns=["Sex"], prefix=["Sex"], drop_first=True)

    df.drop(axis=1, columns="Rings")

    return df


@task(name="val-split", tags=["fails"], retries=3, retry_delay_seconds=60)
def extract_x_y_split(
    df: pd.DataFrame, target: str = "age"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits DataFrame into feature (X) and target (y) sets and splits them into train, test sets.

    Parameters
    -----------
    df : pd.DataFrame
         The input DataFrame containing the dataset.

    target : str (optional)
             The name of the target column in the DataFrame. Default is "age".

    Returns
    --------
    X_train : pd.DataFrame
              The training feature set (X).
    X_test : pd.DataFrame
             The testing feature set (X).
    y_train : pd.Series
              The training target set (y).
    y_test : pd.Series
             The testing target set (y).
    """
    X = df.loc[:, df.columns != target]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test
