import time
train_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

# Script Dependencies: pandas, numpy, tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# [Req] Import params
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

# Import custom made functions
from uno_utils_improve import data_generator, batch_predict, print_duration, clean_arrays, check_array

# Model imports
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    LearningRateScheduler,
    EarlyStopping,
)

# Check tensorflow and GPU
print("Tensorflow Version:")
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

filepath = Path(__file__).resolve().parent  # [Req]

"""
# Notes: 
  - Model.fit initalizes batch before epoch, causing that generator to be off a batch size.
  - Do not use same generator to make predictions... results in index shift that cause v1=v2 error
  - Predictions are underestimates much more often than not... probably because there are lots of
    auc values close to 1 because we only have effective drugs in our dataset and sigmoid has small
    curvature, making extreme values very difficult. If we have lots of ineffective drugs, we will
    have extreme values close to 0 as well. Worth coming up with a more straightened out activation
    function to allow for extreme values more often.
  - power_yj scaler that is made from a different cross-study dataset can cause NaNs from exploding
    or vanishing gradients. This is because the power_yj scaler is not robust to extreme values and
    requires cleaning of the array before storing test scores.
  - Incorporate R2 calculation into the normal loss calculation? How it's done right now requires
    loading and predicting the entire dataset again, which is not efficient.
"""

# ---------------------
# [Req] Parameter lists
# ---------------------
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def warmup_scheduler(epoch, lr, warmup_epochs, initial_lr, max_lr, warmup_type):
    if epoch <= warmup_epochs:
        if warmup_type == "none" or warmup_type == "constant":
            lr = max_lr
        elif warmup_type == "linear":
            lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
        elif warmup_type == "quadratic":
            lr = initial_lr + (max_lr - initial_lr) * ((epoch / warmup_epochs) ** 2)
        elif warmup_type == "exponential":
            lr = initial_lr * ((max_lr / initial_lr) ** (epoch / warmup_epochs))
        else:
            raise ValueError("Invalid warmup type")
    return float(lr)  # Ensure returning a float value


def calculate_sstot(y):
    """
    Calculate the total sum of squares (SStot) using a NumPy array.

    :param y: NumPy array of observed values.
    :return: Total sum of squares (SStot).
    """
    # Calculate the mean of the observed values
    y_mean = np.mean(y)

    # Calculate the total sum of squares
    sstot = np.sum((y - y_mean) ** 2)

    return sstot


class R2Callback(Callback):
    def __init__(self, model, r2_train_generator, r2_val_generator, train_steps, validation_steps, sstot_train, sstot_val, train_debug=False):
        super().__init__()
        self.model = model
        self.train_generator = r2_train_generator
        self.val_generator = r2_val_generator
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.sstot_train = sstot_train
        self.sstot_val = sstot_val
        self.train_debug = train_debug

    def on_epoch_end(self, epoch, logs=None):
        # print("R2 Start")
        r2_train = self.compute_r2(self.train_generator, self.train_steps, self.sstot_train, self.train_debug)
        r2_val = self.compute_r2(self.val_generator, self.validation_steps, self.sstot_val, self.train_debug)
        logs["r2_train"] = r2_train
        logs["r2_val"] = r2_val
        # print("R2 End")

    def compute_r2(self, data_generator, steps, sstot, train_debug):
        y_pred, y_true = batch_predict(self.model, data_generator, steps)
        ssres = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ssres / sstot)
        if train_debug:
            print("\n R2 Calculation:")
            print(f"y_pred Size: {len(y_pred)}")
            print(f"y_true Size: {len(y_true)}")
            print(f"SSres Check: {ssres}")
            print(f"SStot Check: {sstot}")
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
    

def read_architecture(params, hyperparam_space, arch_type):
    # Setup the architecture for cancer, drug, and interaction layers.
    """
    This function is made to allow for different hyperparameter spaces to be used for the architecture,
    allowing for global, block, and layer hyperparameter spaces when performing HPO. Depending on whether
    the hyperparameter space is global, block, or layer, the architecture will read from the global,
    block, or layer hyperparameters from the default model file for the cancer, drug, and interaction layers. 
    This way, the architecture can be defined in a more flexible way, allowing for more complex architectures 
    to be defined.
    """
    layers_size = []
    layers_dropout = []
    layers_activation = []

    if hyperparam_space == "global":
        if arch_type == "canc" or "drug":
            num_layers = 3
        elif arch_type == "interaction":
            num_layers = 5
        layers_size = [1000] * num_layers
        layers_dropout = [params["dropout"]] * num_layers
        layers_activation = [params["activation"]] * num_layers
    elif hyperparam_space == "block":
        num_layers = params[f"{arch_type}_num_layers"]
        arch = params[f"{arch_type}_arch"]
        layers_size = arch
        layers_dropout = [params[f"{arch_type}_dropout"]] * num_layers
        layers_activation = [params[f"{arch_type}_activation"]] * num_layers
    elif hyperparam_space == "layer":
        num_layers = params[f"{arch_type}_num_layers"]
        for i in range(num_layers):
            layers_size.append(params[f"{arch_type}_layer_{i+1}_size"])
            layers_dropout.append(params[f"{arch_type}_layer_{i+1}_dropout"])
            layers_activation.append(params[f"{arch_type}_layer_{i+1}_activation"])

    return num_layers, layers_size, layers_dropout, layers_activation


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
    generator_batch_size = params["generator_batch_size"]
    raw_max_lr = params["raw_max_lr"]
    raw_min_lr = raw_max_lr / 1000
    max_lr = raw_max_lr * batch_size
    min_lr = raw_min_lr * batch_size
    warmup_epochs = params["warmup_epochs"]
    warmup_type = params["warmup_type"]
    initial_lr = raw_max_lr / 1000
    reduce_lr_factor = params["reduce_lr_factor"]
    reduce_lr_patience = params["reduce_lr_patience"]
    early_stopping_patience = params["early_stopping_patience"]
    optimizer = get_optimizer(params["optimizer"], initial_lr)
    train_debug = params["train_debug"]
    train_subset_data = params["train_subset_data"]
    preprocess_subset_data = params["preprocess_subset_data"]


    # Architecture Hyperparams
    hyperparam_space = params["hyperparam_space"]
    print(f"Hyperparam Space: {hyperparam_space}")

    # Read architecture for cancer, drug, and interaction layers
    # Uses the read_architecture function to allow for different hyperparameter spaces
    canc_num_layers, canc_layers_size, canc_layers_dropout, canc_layers_activation = read_architecture(params, hyperparam_space, "canc")
    drug_num_layers, drug_layers_size, drug_layers_dropout, drug_layers_activation = read_architecture(params, hyperparam_space, "drug")
    interaction_num_layers, interaction_layers_size, interaction_layers_dropout, interaction_layers_activation = read_architecture(params, hyperparam_space, "interaction")

    # Final regression layer
    regression_activation = params["regression_activation"]

    # Print architecture in debug mode
    if train_debug:
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
        print("REGRESSION LAYER:")
        print(regression_activation)


    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)

    # Subsetting the data for faster training (debugging)
    if train_subset_data:
        # Define total number of samples to subset (or all for small datasets)
        total_num_samples = 5000
        dataset_proportions = {"Train": 0.8, "Validation": 0.1}
        datasets = {"Train": tr_data, "Validation": vl_data}  # Use a dictionary to store datasets

        for key, dataset in datasets.items():
            proportion = dataset_proportions[key]
            num_samples = int(total_num_samples * proportion)
            
            if num_samples >= dataset.shape[0]:
                print(f"Dataset '{key}' is already small ({dataset.shape[0]} samples). Subsetting is skipped.")
            else:
                print(f"Subsetting '{key}' dataset to {num_samples} samples.")
                datasets[key] = dataset.sample(n=num_samples).reset_index(drop=True)

        # Update variables after subsetting
        tr_data, vl_data = datasets["Train"], datasets["Validation"]

    # Show data in debug mode
    if train_debug:
        print("TRAIN DATA:")
        print(tr_data.head())
        print(tr_data.shape)
        print("")
        print("VAL DATA:")
        print(vl_data.head())
        print(vl_data.shape)


    # Identify the Feature Sets from DataFrame
    num_ge_columns = len([col for col in tr_data.columns if col.startswith('ge')])
    num_md_columns = len([col for col in tr_data.columns if col.startswith('mordred')])

    # Separate input (features) and target and convert to numpy arrays
    # (better for tensorflow)
    x_train = tr_data.iloc[:, 1:].to_numpy()
    y_train = tr_data.iloc[:, 0].to_numpy()
    x_val = vl_data.iloc[:, 1:].to_numpy()
    y_val = vl_data.iloc[:, 0].to_numpy()

    # Slice the input tensor
    all_input = Input(shape=(num_ge_columns + num_md_columns,), name="all_input")
    canc_input = Lambda(lambda x: x[:, :num_ge_columns])(all_input)
    drug_input = Lambda(lambda x: x[:, num_ge_columns:num_ge_columns + num_md_columns])(all_input)

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
    output = Dense(1, activation=regression_activation)(interaction_encoded)  # A single continuous value such as AUC

    # Compile Model
    model = Model(inputs=all_input, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )
    # Observe model if debugging mode
    if train_debug:
        model.summary()


    # Number of batches for data loading and callbacks
    # steps_per_epoch is for grad descent batches while train_steps is for r2_train
    steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
    train_steps = int(np.ceil(len(x_train) / generator_batch_size))
    validation_steps = int(np.ceil(len(x_val) / generator_batch_size))


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
    sstot_train = calculate_sstot(y_train)
    sstot_val = calculate_sstot(y_val)
    # Create generators to calculate r2
    r2_train_gen = data_generator(x_train, y_train, generator_batch_size)
    r2_val_gen = data_generator(x_val, y_val, generator_batch_size)
    # Initialize R2Callback with generators
    r2_callback = R2Callback(
        model=model,
        r2_train_generator=r2_train_gen,
        r2_val_generator=r2_val_gen,
        train_steps=train_steps,
        validation_steps=validation_steps,
        sstot_train=sstot_train,
        sstot_val=sstot_val,
        train_debug=train_debug
    )


    epoch_start_time = time.time()


    # Make separate generators for training (fixing index issue)
    train_gen = data_generator(x_train, y_train, batch_size, shuffle=True, peek=True)
    val_gen = data_generator(x_val, y_val, generator_batch_size, peek=True)

    # Fit model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
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
    # Make sure to make new generator state so no index problem
    val_pred, val_true = batch_predict(model, data_generator(x_val, y_val, generator_batch_size), validation_steps)


    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    # Data must be subsetted in preprocess AND train or neither. Dangerous if parameter changes without running again
    if (train_subset_data and preprocess_subset_data) or (not train_subset_data and not preprocess_subset_data):    
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


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params
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
