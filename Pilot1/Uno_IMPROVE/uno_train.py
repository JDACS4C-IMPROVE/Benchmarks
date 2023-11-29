from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    Callback,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model


from uno_preprocess_improve import model_preproc_params

filepath = Path(__file__).resolve().parent

req_train_args = ["model_arch", "model_outdir",
                  "train_ml_data_dir", "val_ml_data_dir",
                  ]

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = [
    # {"name": "val_data_df",
    #  "default": frm.SUPPRESS,
    #  "type": str,
    #  "help": "Data frame with original validation response data."
    # },
]

# [UNO] Model-specific params (Model: UNO)
model_train_params = [
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",  # TODO. frm. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },
]


def initialize_parameters():
   pass

def run(params):
    pass

def main():
    additional_definitions = model_train_params + \
                             model_preproc_params + \
                             app_train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="graphdrp_default_model.txt",
        additional_definitions=additional_definitions,
        required=req_train_args,
    )
    val_scores = run(params)
    print("\nFinished training UNO model.")


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
