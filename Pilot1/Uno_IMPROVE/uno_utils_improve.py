import numpy as np


def data_generator(x_data, y_data, batch_size, shuffle=False, peek=False, verbose=False):
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    
    if peek:    # Give first batch unshuffled and don't change start index when peeking for training
        end = min(batch_size, num_samples)
        if verbose:
            print(f"Generating peeking batch up to index {end}")
        batch_x = x_data[:end]
        batch_y = y_data[:end]
        peek = False
        yield (batch_x, batch_y)

    while True:    # Loop indefinitely for epochs
        # Shuffle indices at the start of each epoch after the peek, if shuffle is enabled
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Print batch indices if verbose
            if verbose:
                # Warning: calling verbose when shuffling will usually clutter output
                if shuffle:
                    if len(batch_indices) < 64:
                        print(f"Batch indices: {np.sort(batch_indices)}")
                    else:
                        print(f"Printing batch indices would clutter output. Skipped.")
                    print(f"Length: {len(batch_indices)}")
                else:
                    print(f"Generating batch from index {start} to {end}")
                # traceback.print_stack()

            # Generate batches
            batch_x = x_data[batch_indices]
            batch_y = y_data[batch_indices]

            # Yield the current batch
            yield (batch_x, batch_y)


def batch_predict(model, data_generator, steps, flatten=True, verbose=False):
    predictions = []
    true_values = []
    for _ in range(steps):
        # print("Batch Predict get next")
        x, y = next(data_generator)
        pred = model.predict(x, verbose=0)
        if flatten:
            pred = pred.flatten()
            y = y.flatten()
        predictions.extend(pred)
        true_values.extend(y)
        if verbose:
            print("Batch Predict:")
            print(f"Predictions: {len(predictions)}")
            print(f"True: {len(true_values)}")
    return np.array(predictions), np.array(true_values)


def print_duration(activity: str, start_time: float, end_time: float):
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")


def clean_arrays(test_pred, test_true):
    # Initialize clean arrays
    test_pred_clean = test_pred
    test_true_clean = test_true

    # Find NaN indices and remove
    nan_indices = np.where(np.isnan(test_pred))[0]
    test_pred_clean = np.delete(test_pred_clean, nan_indices)
    test_true_clean = np.delete(test_true_clean, nan_indices)

    # Find infinity indices and remove
    inf_indices = np.where(np.isinf(test_pred))[0]
    test_pred_clean = np.delete(test_pred_clean, inf_indices)
    test_true_clean = np.delete(test_true_clean, inf_indices)

    # Print the number and percent of removed indices
    start_len = len(test_pred)
    end_len = len(test_pred_clean)
    print(f"Removed {start_len - end_len} values due to NaN or infinity values.")
    print(f"Removed {100 * (start_len - end_len) / start_len:.3f}% of data due to NaN or infinity values.")

    return test_pred_clean, test_true_clean


def check_array(array):
    # Print shape
    print(f"Shape: {array.shape}")

    # Print the first few values
    print("First few values:", array[:5])

    # Check and print indices/values for NaN values
    nan_indices = np.where(np.isnan(array))[0]
    print("Indices of NaN:", nan_indices[:5])

    # Check and print indices/values for infinity values
    inf_indices = np.where(np.isinf(array))[0]
    print("Indices of infinity:", inf_indices[:5])