""" Preprocessing of raw data to generate datasets for UNO Model. """
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# IMPROVE imports
from improve import framework as frm
# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
from improve import drug_resp_pred as drp  # some funcs from dataloader.py were copied to drp

filepath = Path(__file__).resolve().parent

drp_preproc_params = [
    {"name": "x_data_canc_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "x_data_drug_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data_files",  # imp;
     # "nargs": "+",
     "type": str,
     "help": "List of output files.",
    },
    {"name": "canc_col_name",  # app;
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",  # app;
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]


model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if using landmark genes.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the scaler object.",
    },
]

preprocess_params = model_preproc_params + drp_preproc_params

req_preprocess_args = [ll["name"] for ll in preprocess_params]  # TODO: it seems that all args specifiied to be 'req'. Why?

req_preprocess_args.extend(["y_col_name", "model_outdir"])  # TODO: Does 'req' mean no defaults are specified?


def run(params):
    """ Execute data pre-processing for UNO model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """


def main():
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished UNO pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
