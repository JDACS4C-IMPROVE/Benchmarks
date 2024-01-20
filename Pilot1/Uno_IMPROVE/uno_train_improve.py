import os
import sys
from pathlib import Path
from typing import Dict

# import uno as benchmark
from improve import framework as frm
from improve.metrics import compute_metrics
import candle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    LearningRateScheduler,
    EarlyStopping,
)

print("Tensorflow Version:")
print(tf.__version__)

filepath = Path(__file__).resolve().parent  # [Req]

# Notes: Permanent Dropout?,

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {
        "name": "canc_layers",
        "type": list,
        "default": [1000, 1000, 1000],
        "help": "Cancer feature layers architecture structure.",
    },
    {
        "name": "canc_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third cancer feature layer.",
    },
    {
        "name": "canc_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third cancer feature layer.",
    },
    {
        "name": "drug_layers",
        "type": list,
        "default": [1000, 1000, 1000],
        "help": "Drug feature layers architecture structure.",
    },
    {
        "name": "drug_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first drug feature layer.",
    },
    {
        "name": "drug_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second drug feature layer.",
    },
    {
        "name": "drug_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third drug feature layer.",
    },
    {
        "name": "drug_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first drug feature layer.",
    },
    {
        "name": "drug_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second drug feature layer.",
    },
    {
        "name": "drug_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third drug feature layer.",
    },
    {
        "name": "interaction_layers",
        "type": list,
        "default": [1000, 1000, 1000],
        "help": "Interaction layers architecture structure.",
    },
    {
        "name": "interaction_layer_1_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for first interaction layer.",
    },
    {
        "name": "interaction_layer_2_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for second interaction layer.",
    },
    {
        "name": "interaction_layer_3_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for third interaction layer.",
    },
    {
        "name": "interaction_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first interaction layer.",
    },
    {
        "name": "interaction_layer_2_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for second interaction layer.",
    },
    {
        "name": "interaction_layer_3_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for third interaction layer.",
    },
    {
        "name": "epochs",
        "type": int,
        "default": 150,
        "help": "Number of epochs in training.",
    },
    {
        "name": "batch_size",
        "type": int,
        "default": 32,
        "help": "Batch size for training.",
    },
    {
        "name": "raw_max_lr",
        "type": float,
        "default": 1e-4,
        "help": "Raw maximum learning rate that is scaled based on batch size and warmed up to with warmup scheduler.",
    },
    {
        "name": "log_10_range",
        "type": int,
        "default": 5,
        "help": "Order of magnitude (base 10) difference between raw_max_lr and raw_min_lr. This makes sure that the learning rate reduction does not go too small.",
    },
    {
        "name": "warmup_epochs",
        "type": int,
        "default": 5,
        "help": "Number of warmup epochs in warmup scheduler",
    },
    {
        "name": "reduce_lr_factor",
        "type": float,
        "default": 0.8,
        "help": "Multiplication factor for reducing learning rate after no improvement.",
    },
    {
        "name": "reduce_lr_patience",
        "type": int,
        "default": 3,
        "help": "Patience for reducing learning rate in scheduler",
    },
    {
        "name": "early_stopping_patience",
        "type": int,
        "default": 10,
        "help": "Patience for early stopping the training for no improvement",
    },
]


# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
# req_train_params = ["model_outdir", "train_ml_data_dir", "val_ml_data_dir"]
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def warmup_scheduler(epoch, lr, warmup_epochs, initial_lr, max_lr, warmup_type):
    if epoch <= warmup_epochs:
        if warmup_type == "linear":
            lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
        elif warmup_type == "quadratic":
            lr = initial_lr + (max_lr - initial_lr) * ((epoch / warmup_epochs) ** 2)
        elif warmup_type == "exponential":
            lr = initial_lr * ((max_lr / initial_lr) ** (epoch / warmup_epochs))
        else:
            raise ValueError("Invalid warmup type")
    return float(lr)  # Ensure returning a float value


class R2Callback(Callback):
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data  # Expected to be a tuple of ([inputs], y)
        self.val_data = val_data  # Expected to be a tuple of ([inputs], y)

    def on_epoch_end(self, epoch, logs=None):
        # Unpack the data
        train_inputs, train_y = self.train_data
        val_inputs, val_y = self.val_data

        # Predictions
        y_train_pred = self.model.predict(train_inputs)
        y_val_pred = self.model.predict(val_inputs)

        # R2 Scores
        r2_train = r2_score(train_y, y_train_pred)
        r2_val = r2_score(val_y, y_val_pred)

        # Logging
        logs["r2_train"] = r2_train
        logs["r2_val"] = r2_val
        print(f"Epoch {epoch+1}: R2 train: {r2_train:.4f}, R2 val: {r2_val:.4f}")


def run(params: Dict):
    """Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # Reading hyperparameters
    # ------------------------------------------------------

    # Learning Hyperparams
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    raw_max_lr = params["raw_max_lr"]
    raw_min_lr = raw_max_lr / (10 ** params["lr_log_10_range"])
    normalizer = np.log2(batch_size) + 1
    max_lr = raw_max_lr * normalizer
    min_lr = raw_min_lr * normalizer
    warmup_epochs = params["warmup_epochs"]
    warmup_type = params["warmup_type"]
    initial_lr = min_lr
    reduce_lr_factor = params["reduce_lr_factor"]
    reduce_lr_patience = params["reduce_lr_patience"]
    early_stopping_patience = params["early_stopping_patience"]

    # Architecture Hyperparams
    # Cancer
    canc_num_layers = params["canc_num_layers"]
    canc_layers_size = []
    canc_layers_dropout = []
    canc_layers_activation = []
    for i in range(canc_num_layers):
        canc_layers_size.append(params[f"canc_layer_{i+1}_size"])
        canc_layers_dropout.append(params[f"canc_layer_{i+1}_dropout"])
        canc_layers_activation.append(params[f"canc_layer_{i+1}_activation"])
    print("CANCER LAYERS:")
    print(canc_layers_size, canc_layers_dropout, canc_layers_activation)
    # Drug
    drug_num_layers = params["drug_num_layers"]
    drug_layers_size = []
    drug_layers_dropout = []
    drug_layers_activation = []
    for i in range(drug_num_layers):
        drug_layers_size.append(params[f"drug_layer_{i+1}_size"])
        drug_layers_dropout.append(params[f"drug_layer_{i+1}_dropout"])
        drug_layers_activation.append(params[f"drug_layer_{i+1}_activation"])
    print("DRUG LAYERS:")
    print(drug_layers_size, drug_layers_dropout, drug_layers_activation)
    # Additional Layers (after concatenation)
    interaction_num_layers = params["interaction_num_layers"]
    interaction_layers_size = []
    interaction_layers_dropout = []
    interaction_layers_activation = []
    for i in range(interaction_num_layers):
        interaction_layers_size.append(params[f"interaction_layer_{i+1}_size"])
        interaction_layers_dropout.append(params[f"interaction_layer_{i+1}_dropout"])
        interaction_layers_activation.append(
            params[f"interaction_layer_{i+1}_activation"]
        )
    print("INTERACTION LAYERS:")
    print(
        interaction_layers_size,
        interaction_layers_dropout,
        interaction_layers_activation,
    )

    # Load the data from CSV
    # Set up datadirs
    train_ml_data_dir = params["train_ml_data_dir"]
    train_split_dir = os.path.join(train_ml_data_dir)
    val_ml_data_dir = params["val_ml_data_dir"]
    val_split_dir = os.path.join(val_ml_data_dir)
    test_ml_data_dir = params["test_ml_data_dir"]
    test_split_dir = os.path.join(test_ml_data_dir)
    # Train filepaths
    train_canc_filepath = os.path.join(train_split_dir, "train_x_canc.parquet")
    train_drug_filepath = os.path.join(train_split_dir, "train_x_drug.parquet")
    train_y_filepath = os.path.join(train_split_dir, "train_y_data.parquet")
    # Validation filepaths
    val_canc_filepath = os.path.join(val_split_dir, "val_x_canc.parquet")
    val_drug_filepath = os.path.join(val_split_dir, "val_x_drug.parquet")
    val_y_filepath = os.path.join(val_split_dir, "val_y_data.parquet")
    # Test filepaths
    test_canc_filepath = os.path.join(test_split_dir, "test_x_canc.parquet")
    test_drug_filepath = os.path.join(test_split_dir, "test_x_drug.parquet")
    test_y_filepath = os.path.join(test_split_dir, "test_y_data.parquet")
    if params["debug"]:
        # Load a subset of the data for debugging
        train_canc_info = pd.read_parquet(train_canc_filepath, nrows=1000)
        train_drug_info = pd.read_parquet(train_drug_filepath, nrows=1000)
        y_train = pd.read_parquet(train_y_filepath, nrows=1000)
        # Validation reads
        val_canc_info = pd.read_parquet(val_canc_filepath, nrows=125)
        val_drug_info = pd.read_parquet(val_drug_filepath, nrows=125)
        y_val = pd.read_parquet(val_y_filepath, nrows=125)
        # Test reads
        test_canc_info = pd.read_parquet(test_canc_filepath, nrows=125)
        test_drug_info = pd.read_parquet(test_drug_filepath, nrows=125)
        y_test = pd.read_parquet(test_y_filepath, nrows=125)
    else:
        # Train reads
        train_canc_info = pd.read_parquet(train_canc_filepath)
        train_drug_info = pd.read_parquet(train_drug_filepath)
        y_train = pd.read_parquet(train_y_filepath)
        # Validation reads
        val_canc_info = pd.read_parquet(val_canc_filepath)
        val_drug_info = pd.read_parquet(val_drug_filepath)
        y_val = pd.read_parquet(val_y_filepath)
        # Test reads
        test_canc_info = pd.read_parquet(test_canc_filepath)
        test_drug_info = pd.read_parquet(test_drug_filepath)
        y_test = pd.read_parquet(test_y_filepath)

    # Cancer expression input and encoding layers
    canc_input = Input(shape=(train_canc_info.shape[1],), name="canc_input")
    canc_encoded = canc_input
    for i in range(canc_num_layers):
        canc_encoded = Dense(canc_layers_size[i], activation=canc_layers_activation[i])(
            canc_encoded
        )
        canc_encoded = Dropout(canc_layers_dropout[i])(canc_encoded)

    # Drug expression input and encoding layers
    drug_input = Input(shape=(train_drug_info.shape[1],), name="drug_input")
    drug_encoded = drug_input
    for i in range(drug_num_layers):
        drug_encoded = Dense(drug_layers_size[i], activation=drug_layers_activation[i])(
            drug_encoded
        )
        drug_encoded = Dropout(drug_layers_dropout[i])(drug_encoded)

    # Concatenated input and interaction layers
    interaction_input = Concatenate()([canc_encoded, drug_encoded])
    interaction_encoded = interaction_input
    for i in range(interaction_num_layers):
        interaction_encoded = Dense(
            interaction_layers_size[i], activation=interaction_layers_activation[i]
        )(interaction_encoded)
        interaction_encoded = Dropout(interaction_layers_dropout[i])(
            interaction_encoded
        )

    # Final output layer
    output = Dense(1)(interaction_encoded)  # A single continuous value such as AUC

    # Compile Model
    model = Model(inputs=[canc_input, drug_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss="mean_squared_error",
    )

    # Instantiate the R2 callback with training and validation data
    train_data_for_callback = ([train_canc_info, train_drug_info], y_train)
    val_data_for_callback = ([val_canc_info, val_drug_info], y_val)
    r2_callback = R2Callback(train_data_for_callback, val_data_for_callback)

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_scheduler(
            epoch, model.optimizer.lr, warmup_epochs, initial_lr, max_lr, warmup_type
        )
    )

    # Reduce learing rate callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,  # Number of epochs with no improvement
        mode="min",
        verbose=1,
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    # Training the model
    history = model.fit(
        [train_canc_info, train_drug_info],
        y_train,
        validation_data=([val_canc_info, val_drug_info], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[r2_callback, lr_scheduler, reduce_lr, early_stopping],
    )

    model.save(modelpath)

    # Compute predictions
    val_pred = model.predict([val_canc_info, val_drug_info])
    val_true = y_val.values
    test_pred = model.predict([test_canc_info, test_drug_info])
    test_true = y_test.values

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    # frm.store_predictions_df(
    #     params,
    #     y_true=val_true,
    #     y_pred=val_pred,
    #     stage="val",
    #     outdir=params["model_outdir"],
    # )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        outdir=params["model_outdir"],
        metrics=metrics_list,
    )
    # Compute test scores
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        outdir=params["model_outdir"],
        metrics=metrics_list,
    )

    return


# [Req]
def main(args):
    # [Req]
    additional_definitions = app_train_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        additional_definitions=additional_definitions,
        # required=req_train_params,
        required=None,
    )
    run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
