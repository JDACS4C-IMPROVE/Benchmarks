import time
infer_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# Additional imports
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

# [Req] Imports from preprocess and train scripts
from uno_preprocess_improve import preprocess_params
from uno_train_improve import metrics_list, train_params

# Should we have preprocessing or assumed done before?
# from sklearn.preprocessing import (
#     StandardScaler,
#     MaxAbsScaler,
#     MinMaxScaler,
#     RobustScaler,
# )

filepath = Path(__file__).resolve().parent  # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def print_duration(activity: str, start_time: float, end_time: float):
    """
    activity (str): Description of the activity.
    duration (int): Duration in minutes.
    """
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")


# [Req]
def run(params: Dict):
    """Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    ts_df = pd.read_parquet(Path(params["test_ml_data_dir"]) / test_data_fname)

    y_ts = ts_df[[params["y_col_name"]]]
    ts_df.drop([params["y_col_name"]], axis=1, inplace=True)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"])  # [Req]

    # Load UNO
    model = load_model(modelpath)

    # Predict and flatten to be accepted by store_predictions
    test_pred = model.predict(ts_df).flatten()
    test_true = y_ts.values.flatten()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        outdir=params["infer_outdir"],
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        outdir=params["infer_outdir"],
        metrics=metrics_list,
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_params,
        required=None,
    )
    test_scores = run(params)
    infer_end_time = time.time()
    print_duration("Infering", infer_start_time, infer_end_time)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])