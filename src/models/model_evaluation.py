import numpy as np
import pandas as pd

import os
import pickle
import json

import src.utils as utils

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# logging configure
logger = utils.configure_logger(__name__, log_file="predict_model.log")

# load model
def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# load data
def load_test_data() -> pd.DataFrame:
    try:
        logger.debug("Loading testing Data")
        data_path = os.path.join("data", "features")
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        return test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":

    # load model
    model = load_model()

    # load data
    test_data = load_test_data()

    if test_data.empty:
        raise ValueError("Data is empty")
    
    # Split X and y
    X_test = test_data.drop("sentiment", axis=1)
    y_test = test_data["sentiment"]

    # predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

    # storing metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }

    # save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logger.info("Metrics saved successfully")
    