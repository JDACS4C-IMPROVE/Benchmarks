import numpy as np
import traceback


def data_generator(x_data, y_data, batch_size, verbose=False):
    num_samples = len(x_data)
    
    while True:  # Loop indefinitely for epochs
        for offset in range(0, num_samples, batch_size):
            # Calculate end of the current batch
            end = min(offset + batch_size, num_samples)
            
            # Print batch indices if verbose
            if verbose:
                print(f"Generating batch from index {offset} to {end}")
                # traceback.print_stack()

            # Generate batches
            batch_x = x_data[offset:end]
            batch_y = y_data[offset:end]

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