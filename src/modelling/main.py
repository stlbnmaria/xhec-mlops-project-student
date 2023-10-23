# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from preprocessing import extract_x_y_split, read_data, transform_data
from training import train_model
from utils import save_pickle


def main(trainset_path: Path, output_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Parameters
    -------
    trainset_path : path
                    Path of the train data.
    output_path : path
                  Path to which the model is saved as a pickle.

    Returns
    -------
    None : None
    """
    # Read data
    data = read_data(trainset_path)
    # Preprocess data
    trans_data = transform_data(data)
    X_train, X_test, y_train, y_test = extract_x_y_split(trans_data)
    # Train model
    model = train_model(X_train, y_train)
    # Pickle model --> The model should be saved in pkl format the
    # `src/web_service/local_objects` folder
    save_pickle(output_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    parser.add_argument("output_path", type=str, help="Path where the pickle model is saved")
    args = parser.parse_args()
    main(args.trainset_path)
