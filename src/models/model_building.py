import numpy as np
import pandas as pd

import os
import pickle

import src.utils as utils

from sklearn.ensemble import GradientBoostingClassifier

# logging configure
logger = utils.configure_logger(__name__, log_file="train_model.log")


def load_data() -> pd.DataFrame:
    try:
        logger.debug("Loading training Data")
        data_path = os.path.join("data", "features")
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        return train_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":

    # load params
    params = utils.load_params("params.yaml", section="model_building", logger=logger)
    # load data
    train_data = load_data()

    # split X and y
    X_train = train_data.drop("sentiment", axis=1)
    y_train = train_data["sentiment"]

    # train model
    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=0.1,
        max_depth=params["max_depth"],
    )
    logger.info("Training Model")
    model.fit(X_train, y_train)

    # save model
    model_path = os.path.join("models")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("Model saved successfully")
