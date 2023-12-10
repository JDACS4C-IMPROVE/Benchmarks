import os
import numpy as np
import pandas as pd
import tensorflow as tf
import uno as benchmark
import candle
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


# Notes: Permanent Dropout?,


def initialize_parameters(default_model="uno_default_model.txt"):
    # Build benchmark object
    UNO = candle.Benchmark(
        benchmark.file_path,
        default_model,
        "keras",
        prog="uno_improve",
        desc="Build neural network based models to predict tumor response to single and paired drugs.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(UNO)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def warmup_scheduler(epoch, lr, warmup_epochs, initial_lr, max_lr):
    if epoch <= warmup_epochs:
        # Linear warmup
        # lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
        # Quadratic warmup
        lr = initial_lr + (max_lr - initial_lr) * ((epoch / warmup_epochs) ** 2)
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


def run(params):
    # Learning Hyperparams
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    raw_max_lr = params["raw_max_lr"]
    raw_min_lr = raw_max_lr / (10 ** params["log_10_range"])
    normalizer = np.log2(batch_size) + 1
    max_lr = raw_max_lr * normalizer
    min_lr = raw_min_lr * normalizer
    warmup_epochs = params["warmup_epochs"]
    initial_lr = min_lr
    reduce_lr_factor = params["reduce_lr_factor"]
    reduce_lr_patience = params["reduce_lr_patience"]
    early_stopping_patience = params["early_stopping_patience"]

    # Architecture Hyperparams
    # Gene
    gene_layers = params["gene_layers"]
    gene_layers_dropout = []
    gene_layers_activation = []
    for i in range(len(gene_layers)):
        dropout_name = f"gene_layer_{i+1}_dropout"
        gene_layers_dropout.append(params[dropout_name])
        activation_name = f"gene_layer_{i+1}_activation"
        gene_layers_activation.append(params[activation_name])
    # Drug
    drug_layers = params["drug_layers"]
    drug_layers_dropout = []
    drug_layers_activation = []
    for i in range(len(drug_layers)):
        dropout_name = f"drug_layer_{i+1}_dropout"
        drug_layers_dropout.append(params[dropout_name])
        activation_name = f"drug_layer_{i+1}_activation"
        drug_layers_activation.append(params[activation_name])
    # Additional Layers (after concatenation)
    interaction_layers = params["interaction_layers"]
    interaction_layers_dropout = []
    interaction_layers_activation = []
    for i in range(len(interaction_layers)):
        dropout_name = f"interaction_layer_{i+1}_dropout"
        interaction_layers_dropout.append(params[dropout_name])
        activation_name = f"interaction_layer_{i+1}_activation"
        interaction_layers_activation.append(params[activation_name])

    # Load the data from CSV
    # Set up datadirs
    candle_data_dir = os.environ.get("CANDLE_DATA_DIR")
    train_ml_data_dir = params["train_ml_data_dir"]
    train_split_dir = os.path.join(candle_data_dir, train_ml_data_dir)
    val_ml_data_dir = params["val_ml_data_dir"]
    val_split_dir = os.path.join(candle_data_dir, val_ml_data_dir)
    # Train filepaths
    train_canc_filepath = os.path.join(train_split_dir, "train_x_canc.csv")
    train_drug_filepath = os.path.join(train_split_dir, "train_x_drug.csv")
    train_y_filepath = os.path.join(train_split_dir, "train_y_data.csv")
    # Validation filepaths
    val_canc_filepath = os.path.join(val_split_dir, "val_x_canc.csv")
    val_drug_filepath = os.path.join(val_split_dir, "val_x_drug.csv")
    val_y_filepath = os.path.join(val_split_dir, "val_y_data.csv")
    # Train reads
    train_gene_info = pd.read_csv(train_canc_filepath)
    train_drug_info = pd.read_csv(train_drug_filepath)
    y_train = pd.read_csv(train_y_filepath)
    # Validation reads
    val_gene_info = pd.read_csv(val_canc_filepath)
    val_drug_info = pd.read_csv(val_drug_filepath)
    y_val = pd.read_csv(val_y_filepath)

    # Gene expression input and encoding layers
    gene_input = Input(shape=(train_gene_info.shape[1],), name="gene_input")
    gene_encoded = gene_input
    for i in range(len(gene_layers)):
        gene_encoded = Dense(gene_layers[i], activation=gene_layers_activation[i])(
            gene_encoded
        )
        gene_encoded = Dropout(gene_layers_dropout[i])(gene_encoded)

    # Drug expression input and encoding layers
    drug_input = Input(shape=(train_drug_info.shape[1],), name="drug_input")
    drug_encoded = drug_input
    for i in range(len(drug_layers)):
        drug_encoded = Dense(drug_layers[i], activation=drug_layers_activation[i])(
            drug_encoded
        )
        drug_encoded = Dropout(drug_layers_dropout[i])(drug_encoded)

    # Concatenated input and interaction layers
    interaction_input = Concatenate()([gene_encoded, drug_encoded])
    interaction_encoded = interaction_input
    for i in range(len(interaction_layers)):
        interaction_encoded = Dense(
            interaction_layers[i], activation=interaction_layers_activation[i]
        )(interaction_encoded)
        interaction_encoded = Dropout(interaction_layers_dropout[i])(
            interaction_encoded
        )

    # Final output layer
    output = Dense(1)(interaction_encoded)  # A single continuous value such as AUC

    # Compile Model
    model = Model(inputs=[gene_input, drug_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss="mean_squared_error",
    )

    # Instantiate the R2 callback with training and validation data
    train_data_for_callback = ([train_gene_info, train_drug_info], y_train)
    val_data_for_callback = ([val_gene_info, val_drug_info], y_val)
    r2_callback = R2Callback(train_data_for_callback, val_data_for_callback)

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_scheduler(
            epoch, model.optimizer.lr, warmup_epochs, initial_lr, max_lr
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
        [train_gene_info, train_drug_info],
        y_train,
        validation_data=([val_gene_info, val_drug_info], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[r2_callback, lr_scheduler, reduce_lr, early_stopping],
    )

    # Evaluate the model
    print(R2Callback(train_data_for_callback, val_data_for_callback))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
