import numpy as np
import pandas as pd
import tensorflow as tf
import uno as benchmark
import candle
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    LearningRateScheduler,
)


# Notes: Permanent Dropout?,


def initialize_parameters(default_model="uno_default_model.txt"):
    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(
        benchmark.file_path,
        default_model,
        "keras",
        prog="uno_baseline",
        desc="Build neural network based models to predict tumor response to single and paired drugs.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(unoBmk)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


warmup_epochs = 20
initial_lr = 1e-5
max_lr = 1e-3


def warmup_scheduler(epoch, lr, warmup_epochs=5, initial_lr=1e-4, max_lr=1e-3):
    if epoch < warmup_epochs:  # Make this depend on batch size at some point
        lr = initial_lr + (max_lr - initial_lr) * epoch / warmup_epochs
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
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)

    # Load the data from CSV
    filepath = "/mnt/c/Users/rylie/Coding/UNO/Benchmarks/Pilot1/Uno_IMPROVE/ml_data/new_processed_data.csv"
    df = pd.read_csv(filepath, header=[0, 1])

    # Separate the features and target
    X = df.drop(("response_values", "AUC"), axis=1)
    Y = df[("response_values", "AUC")]

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Split gene and drug information
    train_gene_info = X_train.loc[:, "gene_info"]
    train_drug_info = X_train.loc[:, "drug_info"]
    val_gene_info = X_val.loc[:, "gene_info"]
    val_drug_info = X_val.loc[:, "drug_info"]

    # Gene expression input and encoding layers
    gene_input = Input(shape=(train_gene_info.shape[1],), name="gene_input")
    gene_encoded = Dense(1024, activation="relu")(gene_input)
    gene_encoded = Dropout(0.1)(gene_encoded)
    gene_encoded = Dense(512, activation="relu")(gene_encoded)
    gene_encoded = Dropout(0.1)(gene_encoded)
    gene_encoded = Dense(256, activation="relu")(gene_encoded)
    gene_encoded = Dropout(0.1)(gene_encoded)

    # Drug expression input and encoding layers
    drug_input = Input(shape=(train_drug_info.shape[1],), name="drug_input")
    drug_encoded = Dense(1024, activation="relu")(drug_input)
    drug_encoded = Dropout(0.1)(drug_encoded)
    drug_encoded = Dense(512, activation="relu")(drug_encoded)
    drug_encoded = Dropout(0.1)(drug_encoded)
    drug_encoded = Dense(256, activation="relu")(drug_encoded)
    drug_encoded = Dropout(0.1)(drug_encoded)

    # Concatenate Encodings
    concatenated = Concatenate()([gene_encoded, drug_encoded])

    # Additional layers
    x = Dense(512, activation="relu")(concatenated)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(1)(x)  # Assuming a single continuous value for AUC

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
        monitor="val_loss", factor=0.9, patience=5, min_lr=1e-6
    )

    # Training the model
    history = model.fit(
        [train_gene_info, train_drug_info],
        y_train,
        validation_data=([val_gene_info, val_drug_info], y_val),
        epochs=150,
        batch_size=32,
        callbacks=[r2_callback, lr_scheduler, reduce_lr],
    )

    # Evaluate the model
    print(R2Callback(train_data_for_callback, val_data_for_callback))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
