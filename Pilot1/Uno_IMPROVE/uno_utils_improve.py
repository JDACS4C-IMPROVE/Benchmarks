import numpy as np
import traceback


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