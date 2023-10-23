# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import mlflow
from predicting import predict
from prefect import flow, serve
from preprocessing import extract_x_y_split, read_data, transform_data
from training import train_model
from utils import evaluate_model, save_pickle

from config.config import DATA_PATH, MODEL_PATH


@flow(retries=3, retry_delay_seconds=5, log_prints=True)
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
    # validate that mlflow runs locally
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

    # set experiment
    mlflow.set_experiment("xgb")

    with mlflow.start_run() as run:
        # get unique identifier of MLflow run
        run_id = run.info.run_id

        # set tags
        mlflow.set_tag("Task_type", "Regression")

        # Read data
        data = read_data(trainset_path)

        # Preprocess data
        trans_data = transform_data(data)
        X_train, X_test, y_train, y_test = extract_x_y_split(trans_data)
        mlflow.log_param("X_train_size", X_train.shape[0])
        mlflow.log_param("X_test_size", X_test.shape[0])

        # train model
        model = train_model(X_train, y_train)
        mlflow.log_params(model.get_params())

        # make prediction on X_test
        y_pred = predict(X_test, model)

        # evaluate on y_test
        mse, r2 = evaluate_model(y_test, y_pred)
        mlflow.log_metric("test_rmse", mse)
        mlflow.log_metric("test_r2", r2)

        # Log your model
        mlflow.xgboost.log_model(model, "model")

        # Register your model
        mlflow.register_model(f"runs:/{run_id}/model", "xgb_regressor")

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

    main_deploy = main.to_deployment(
        name="train",
        cron="0 0 1 * *",  # run once a month on the first day at midnight
        parameters={
            "trainset_path": args.trainset_path,
            "model_path": args.model_path,
        },
    )
    serve(main_deploy)
