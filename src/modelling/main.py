# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from preprocessing import extract_x_y_split, read_data, transform_data
from training import train_model
from utils import save_pickle

from config.config import DATA_PATH, MODEL_PATH


def main(trainset_path: Path, model_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Parameters
    -------
    trainset_path : Path
                    Path of the train data.
    model_path : Path
                  Path to which the model is saved as a pickle.

    Returns
    -------
    None : None
    """
    # Read data
    data = read_data(trainset_path)

    # Preprocess data
    trans_data = transform_data(data)
    X_train, _, y_train, _ = extract_x_y_split(trans_data)

    # Train model
    model = train_model(X_train, y_train)

    # Pickle model
    save_pickle(model_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument(
        "--trainset_path", type=Path, help="Path to the training set", default=DATA_PATH
    )
    parser.add_argument(
        "--model_path", type=Path, help="Path where the pickle model is saved", default=MODEL_PATH
    )
    args = parser.parse_args()
    print(args.trainset_path)
    main(trainset_path=args.trainset_path, model_path=args.model_path)
