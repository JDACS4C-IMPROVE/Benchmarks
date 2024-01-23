import time
infer_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

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

preprocess_params = []
train_params = []

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
    # test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    # x_test_data = pd.read_csv(Path(params["test_ml_data_dir"]) / test_data_fname)

    # Test filepaths
    test_canc_filepath = os.path.join(params["test_ml_data_dir"], "test_x_canc.parquet")
    test_drug_filepath = os.path.join(params["test_ml_data_dir"], "test_x_drug.parquet")
    test_y_filepath = os.path.join(params["test_ml_data_dir"], "test_y_data.parquet")
    # Test reads
    test_canc_info = pd.read_parquet(test_canc_filepath)
    test_drug_info = pd.read_parquet(test_drug_filepath)
    y_test = pd.read_parquet(test_y_filepath)

    # fea_list = ["ge", "mordred"]
    # fea_sep = "."

    # Test data
    # xte = extract_subset_fea(test_data, fea_list=fea_list, fea_sep=fea_sep)
    # yte = test_data[[params["y_col_name"]]]
    # print("xte:", xte.shape)
    # print("yte:", yte.shape)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"])  # [Req]

    # Load UNO
    model = load_model(modelpath)

    # Predict
    test_pred = model.predict([test_canc_info, test_drug_info])
    test_true = y_test.values
    print("Type of test_true:", type(test_true))
    print("Type of test_pred:", type(test_pred))
    if isinstance(test_true, np.ndarray):
        print("Shape of test_true:", test_true.shape)
    if isinstance(test_pred, np.ndarray):
        print("Shape of test_pred:", test_pred.shape)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    # frm.store_predictions_df(
    #     params,
    #     y_true=test_true,
    #     y_pred=test_pred,
    #     stage="test",
    #     outdir=params["infer_outdir"],
    # )

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
    print(f"Infer Time = {infer_end_time - infer_start_time} seconds")
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])