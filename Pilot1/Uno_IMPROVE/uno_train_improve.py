import time
train_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

from improve import framework as frm
from improve.metrics import compute_metrics
from uno_utils_improve import data_generator, batch_predict, print_duration
import candle

import numpy as np
import pandas as pd
import tensorflow as tf
# Configure GPU memory growth for big datasets
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    LearningRateScheduler,
    EarlyStopping,
)

print("Tensorflow Version:")
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

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

# 2. Model-specific params (Model: UNO)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {
        "name": "canc_num_layers",
        "type": int,
        "default": 3,
        "help": """
                Number of cancer feature layers. The
                script reads layer sizes, dropouts, and
                activation up to number of layers specified.
                """,
    },
    {
        "name": "canc_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first cancer feature layer.",
    },
    {
        "name": "canc_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second cancer feature layer.",
    },
    {
        "name": "canc_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third cancer feature layer.",
    },
    {
        "name": "canc_layer_4_size",
        "type": int,
        "default": 512,
        "help": "Size of fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_size",
        "type": int,
        "default": 256,
        "help": "Size of fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_size",
        "type": int,
        "default": 128,
        "help": "Size of sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_size",
        "type": int,
        "default": 64,
        "help": "Size of seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_size",
        "type": int,
        "default": 32,
        "help": "Size of eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_size",
        "type": int,
        "default": 32,
        "help": "Size of ninth cancer feature layer.",
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
        "name": "canc_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth cancer feature layer.",
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
        "name": "canc_layer_4_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fourth cancer feature layer.",
    },
    {
        "name": "canc_layer_5_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fifth cancer feature layer.",
    },
    {
        "name": "canc_layer_6_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for sixth cancer feature layer.",
    },
    {
        "name": "canc_layer_7_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for seventh cancer feature layer.",
    },
    {
        "name": "canc_layer_8_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for eighth cancer feature layer.",
    },
    {
        "name": "canc_layer_9_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for ninth cancer feature layer.",
    },
    {
        "name": "drug_num_layers",
        "type": int,
        "default": 3,
        "help": """
                Number of drug feature layers. The script
                reads layer sizes, dropouts, and activation
                up to number of layers specified.
                """,
    },
    {
        "name": "drug_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first drug feature layer.",
    },
    {
        "name": "drug_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second drug feature layer.",
    },
    {
        "name": "drug_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third drug feature layer.",
    },
    {
        "name": "drug_layer_4_size",
        "type": int,
        "default": 512,
        "help": "Size of fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_size",
        "type": int,
        "default": 256,
        "help": "Size of fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_size",
        "type": int,
        "default": 128,
        "help": "Size of sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_size",
        "type": int,
        "default": 64,
        "help": "Size of seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_size",
        "type": int,
        "default": 32,
        "help": "Size of eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_size",
        "type": int,
        "default": 32,
        "help": "Size of ninth drug feature layer.",
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
        "name": "drug_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth drug feature layer.",
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
        "name": "drug_layer_4_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fourth drug feature layer.",
    },
    {
        "name": "drug_layer_5_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for fifth drug feature layer.",
    },
    {
        "name": "drug_layer_6_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for sixth drug feature layer.",
    },
    {
        "name": "drug_layer_7_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for seventh drug feature layer.",
    },
    {
        "name": "drug_layer_8_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for eighth drug feature layer.",
    },
    {
        "name": "drug_layer_9_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for ninth drug feature layer.",
    },
    {
        "name": "interaction_num_layers",
        "type": int,
        "default": 5,
        "help": """
                Number of interaction feature layers. The
                script reads layer sizes, dropouts, and
                activation up to number of layers specified.
                """,
    },
    {
        "name": "interaction_layer_1_size",
        "type": int,
        "default": 1000,
        "help": "Size of first interaction layer.",
    },
    {
        "name": "interaction_layer_2_size",
        "type": int,
        "default": 1000,
        "help": "Size of second interaction layer.",
    },
    {
        "name": "interaction_layer_3_size",
        "type": int,
        "default": 1000,
        "help": "Size of third interaction layer.",
    },
    {
        "name": "interaction_layer_4_size",
        "type": int,
        "default": 1000,
        "help": "Size of fourth interaction layer.",
    },
    {
        "name": "interaction_layer_5_size",
        "type": int,
        "default": 1000,
        "help": "Size of fifth interaction layer.",
    },
    {
        "name": "interaction_layer_6_size",
        "type": int,
        "default": 512,
        "help": "Size of sixth interaction layer.",
    },
    {
        "name": "interaction_layer_7_size",
        "type": int,
        "default": 256,
        "help": "Size of seventh interaction layer.",
    },
    {
        "name": "interaction_layer_8_size",
        "type": int,
        "default": 128,
        "help": "Size of eighth interaction layer.",
    },
    {
        "name": "interaction_layer_9_size",
        "type": int,
        "default": 64,
        "help": "Size of ninth interaction layer.",
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
        "name": "interaction_layer_4_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fourth interaction layer.",
    },
    {
        "name": "interaction_layer_5_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for fifth interaction layer.",
    },
    {
        "name": "interaction_layer_6_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for sixth interaction layer.",
    },
    {
        "name": "interaction_layer_7_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for seventh interaction layer.",
    },
    {
        "name": "interaction_layer_8_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for eighth interaction layer.",
    },
    {
        "name": "interaction_layer_9_dropout",
        "type": float,
        "default": 0.1,
        "help": "Dropout for ninth interaction layer.",
    },
    {
        "name": "interaction_layer_1_activation",
        "type": str,
        "default": "relu",
        "help": "Activation for first interaction layer.",
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
        "name": "val_batch",
        "type": int,
        "default": 256,
        "help": "Validation batch size.",
    },
    {
        "name": "raw_max_lr",
        "type": float,
        "default": 1e-6,
        "help": "Raw maximum learning rate that is scaled according to batch size.",
    },
    {
        "name": "lr_log_10_range",
        "type": int,
        "default": 3,
        "help": """
                Log 10 range for min learning from initial 
                learning rate. Used in warmup and plateau.
                """,
    },
    {
        "name": "warmup_epochs",
        "type": int,
        "default": 5,
        "help": "Number of warmup epochs.",
    },
    {
        "name": "warmup_type",
        "type": str,
        "default": "quadratic",
        "help": "Type of warmup for learning rate.",
    },
    {
        "name": "reduce_lr_patience",
        "type": int,
        "default": 3,
        "help": "Patience epochs for reducing learning rate.",
    },
    {
        "name": "reduce_lr_factor",
        "type": float,
        "default": 0.5,
        "help": "Factor for reducing learning rate after plateau.",
    },
    {
        "name": "loss",
        "type": str,
        "default": "mse",
        "help": "Loss function to be used.",
    },
    {
        "name": "early_stop_metric",
        "type": str,
        "default": "mse",
        "help": "Loss function for early stopping",
    },
    {
        "name": "early_stopping_patience",
        "type": int,
        "default": 15,
        "help": "Patience for early stopping training after no improvement",
    },
]


# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
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
    def __init__(self, train_data_generator, val_data_generator, steps_per_epoch, validation_steps):
        super().__init__()
        self.train_data_generator = train_data_generator
        self.val_data_generator = val_data_generator
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        r2_train = self.compute_r2(self.train_data_generator, self.steps_per_epoch)
        r2_val = self.compute_r2(self.val_data_generator, self.validation_steps)
        
        logs["r2_train"] = r2_train
        logs["r2_val"] = r2_val

    def compute_r2(self, data_generator, steps):
        y_pred, y_true = batch_predict(self.model, data_generator, steps, flatten=True)
        r2 = r2_score(y_true, y_pred)
        return r2
    

def get_optimizer(optimizer_name, initial_lr):
    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=initial_lr)
    elif optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=initial_lr)
    elif optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=initial_lr)
    elif optimizer_name == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=initial_lr)
    elif optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=initial_lr)
    elif optimizer_name == "Adamax":
        return tf.keras.optimizers.Adamax(learning_rate=initial_lr)
    elif optimizer_name == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate=initial_lr)
    elif optimizer_name == "Ftrl":
        return tf.keras.optimizers.Ftrl(learning_rate=initial_lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not recognized")


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
    val_batch = params["val_batch"]
    test_batch = params["test_batch"]
    raw_max_lr = params["raw_max_lr"]
    raw_min_lr = raw_max_lr / (10 ** params["lr_log_10_range"])
    max_lr = raw_max_lr * batch_size
    min_lr = raw_min_lr * batch_size
    warmup_epochs = params["warmup_epochs"]
    warmup_type = params["warmup_type"]
    initial_lr = min_lr
    reduce_lr_factor = params["reduce_lr_factor"]
    reduce_lr_patience = params["reduce_lr_patience"]
    early_stopping_patience = params["early_stopping_patience"]
    optimizer = get_optimizer(params["optimizer"], initial_lr)

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
    # Drug
    drug_num_layers = params["drug_num_layers"]
    drug_layers_size = []
    drug_layers_dropout = []
    drug_layers_activation = []
    for i in range(drug_num_layers):
        drug_layers_size.append(params[f"drug_layer_{i+1}_size"])
        drug_layers_dropout.append(params[f"drug_layer_{i+1}_dropout"])
        drug_layers_activation.append(params[f"drug_layer_{i+1}_activation"])
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
    # Print architecture in debug mode
    if params["train_debug"]:
        print("CANCER LAYERS:")
        print(canc_layers_size, canc_layers_dropout, canc_layers_activation)
        print("DRUG LAYERS:")
        print(drug_layers_size, drug_layers_dropout, drug_layers_activation)
        print("INTERACTION LAYERS:")
        print(
            interaction_layers_size,
            interaction_layers_dropout,
            interaction_layers_activation,
        )

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)
    ts_data = pd.read_parquet(Path(params["test_ml_data_dir"])/test_data_fname)

    # Subsetting the data for faster training if desired
    if params["train_subset_data"]:
        # Subset 5000 samples (or all for small datasets)
        total_num_samples = 5000
        dataset_proportions = {"train": 0.8, "validation": 0.1, "test": 0.1}
        dataset_size = tr_data.shape[0] + vl_data.shape[0] + ts_data.shape[0]
        if total_num_samples >= dataset_size:
            print(f"Small Dataset of Size {dataset_size}. "
            f"Subsetting to {total_num_samples} Is Skipped")
        else:
            num_samples = {dataset: int(total_num_samples * proportion) for dataset, proportion in dataset_proportions.items()}
            print(f"Subsetting Data To: {num_samples}")
            # Subsetting the datasets
            tr_data = tr_data.sample(n=num_samples["train"]).reset_index(drop=True)
            vl_data = vl_data.sample(n=num_samples["validation"]).reset_index(drop=True)
            ts_data = ts_data.sample(n=num_samples["test"]).reset_index(drop=True) 

    # Show data in debug mode
    if params["train_debug"]:
        print("TRAIN DATA:")
        print(tr_data.head())
        print(tr_data.shape)
        print("")
        print("VAL DATA:")
        print(vl_data.head())
        print(vl_data.shape)
        print("")
        print("TEST DATA:")        
        print(ts_data.head())
        print(ts_data.shape)
        print("")

    # Identify the Feature Sets from DataFrame
    num_ge_columns = len([col for col in tr_data.columns if col.startswith('ge')])
    num_md_columns = len([col for col in tr_data.columns if col.startswith('mordred')])

    # Separate input (features) and target and convert to numpy arrays
    # (better for tensorflow)
    x_train = tr_data.iloc[:, 1:].to_numpy()
    y_train = tr_data.iloc[:, 0].to_numpy()
    x_val = vl_data.iloc[:, 1:].to_numpy()
    y_val = vl_data.iloc[:, 0].to_numpy()
    x_test = ts_data.iloc[:, 1:].to_numpy()
    y_test = ts_data.iloc[:, 0].to_numpy()

    # Slice the input tensor
    all_input = Input(shape=(num_ge_columns + num_md_columns,), name="all_input")
    canc_input = Lambda(lambda x: x[:, :num_ge_columns])(all_input)
    drug_input = Lambda(lambda x: x[:, num_ge_columns:num_ge_columns + num_md_columns])(all_input)

    # Create batch generators to help with memory issues on large datasets
    train_data = data_generator(x_train, y_train, batch_size)
    val_data_generator = data_generator(x_val, y_val, val_batch)
    test_data_generator = data_generator(x_test, y_test, test_batch)
    # Number of batches in training/val/test
    steps_per_epoch = int(np.ceil(len(x_train) // batch_size))
    validation_steps = int(np.ceil(len(x_val) / val_batch))
    test_steps = int(np.ceil(len(x_test) / test_batch))

    # Cancer expression input and encoding layers
    canc_encoded = canc_input
    for i in range(canc_num_layers):
        canc_encoded = Dense(canc_layers_size[i], activation=canc_layers_activation[i])(canc_encoded)
        canc_encoded = Dropout(canc_layers_dropout[i])(canc_encoded)

    # Drug expression input and encoding layers
    drug_encoded = drug_input
    for i in range(drug_num_layers):
        drug_encoded = Dense(drug_layers_size[i], activation=drug_layers_activation[i])(drug_encoded)
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
    model = Model(inputs=all_input, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    
    # Instantiate callbacks

    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_scheduler(
            epoch, model.optimizer.lr, warmup_epochs, initial_lr, max_lr, warmup_type
        )
    )

    # Reduce learing rate
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    # R2
    r2_callback = R2Callback(train_data, val_data_generator, steps_per_epoch, validation_steps)


    epoch_start_time = time.time()

    # Training the model
    history = model.fit(
        train_data,
        validation_data=val_data_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[r2_callback, lr_scheduler, reduce_lr, early_stopping],
    )

    # Calculate the time per epoch
    epoch_end_time = time.time()
    total_epochs = len(history.history['loss'])  # Get the actual number of epochs
    global time_per_epoch 
    time_per_epoch = (epoch_end_time - epoch_start_time) / total_epochs

    # Save model
    model.save(modelpath)

    # Batch prediction (and flatten inside function)
    val_pred, val_true = batch_predict(model, val_data_generator, validation_steps)
    test_pred, test_true = batch_predict(model, test_data_generator, test_steps)

    # Print the shapes to verify they match
    print("Validation predictions shape:", val_pred.shape)
    print("Validation true values shape:", val_true.shape)
    print("Test predictions shape:", test_pred.shape)
    print("Test true values shape:", test_true.shape)


    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    if not params["train_subset_data"]:   # temporary fix. incompatible function with subsetting
        frm.store_predictions_df(
            params,
            y_true=val_true,
            y_pred=val_pred,
            stage="val",
            outdir=params["model_outdir"],
        )

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
    train_end_time = time.time()
    print_duration("One epoch", 0, time_per_epoch)
    print_duration("Total Training", train_start_time, train_end_time)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
