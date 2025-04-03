import numpy as np
import pandas as pd

import os
import pickle
from dvclive import Live

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

    # load params
    params = utils.load_params("params.yaml", section="all", logger=logger)

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

    with Live(save_dvc_exp=True) as live:
        live.log_metric("accuracy", accuracy)
        live.log_metric("precision", precision)
        live.log_metric("recall", recall)
        live.log_metric("roc_auc", roc_auc)

        for module, values in params.items():
            for key, value in values.items():
                live.log_metric(f"{module}_{key}", value)
