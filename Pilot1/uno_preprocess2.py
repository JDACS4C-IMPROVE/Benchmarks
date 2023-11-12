""" Preprocessing of raw data to generate datasets for UNO Model. """
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# IMPROVE imports
import sys
sys.path.append('/mnt/c/Users/rylie/Coding/Uno_IMPROVE/IMPROVE')
from improve import framework as frm
# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
# from improve import drug_resp_pred as drp  # some funcs from dataloader.py were copied to drp

filepath = Path(__file__).resolve().parent

drp_preproc_params = [
    {"name": "x_data_canc_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of cancer feature files.",
    },
    {"name": "x_data_drug_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of drug feature files.",
    },
    {"name": "y_data_files",  # imp;
     # "nargs": "+",
     "type": str,
     "help": "List of response files.",
    },
    {"name": "canc_col_name",  # app;
     "default": "CancID",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",  # app;
     "default": "DrugID",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]

model_preproc_params = [
    {"name": "genetic_feature_scaling",
     # "nargs": "+",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "maabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "drug_feature_scaling",
     # "nargs": "+",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "maabs", "robust"],
     "help": "Scaler for drug data.",
    },
]

# Define a dictionary mapping parameter names to scaler classes
scalers = {
    'std': StandardScaler,
    'maabs': MaxAbsScaler,
    'maabs': MaxAbsScaler,
    'robust': RobustScaler
}

def get_scaler(scaler_param):
    if scaler_param == 'std':
        return StandardScaler()
    elif scaler_param == 'minmax':
        return MinMaxScaler()
    elif scaler_param == 'maxabs':
        return MaxAbsScaler()
    elif scaler_param == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type provided: {scaler_param}")

def merge_dataframes_on_common_id(dataframes_dict, common_id_fields):
    # Start with an empty DataFrame for merging results
    merged_df = pd.DataFrame()

    # Iterate over the items in dataframes_dict to merge them
    for key, df in dataframes_dict.items():
        # If merged_df is empty, initialize it with the first DataFrame
        if merged_df.empty:
            merged_df = df
        else:
            # Determine which common ID to use based on the columns available
            common_columns = list(set(merged_df.columns) & set(df.columns) & set(common_id_fields))
            # Merge the DataFrames on the common columns
            if common_columns:
                merged_df = merged_df.merge(df, on=common_columns, how='inner')
            else:
                print(f"No common columns to merge on for {key}.")
    return merged_df

preprocess_params = model_preproc_params + drp_preproc_params

req_preprocess_args = [ll["name"] for ll in drp_preproc_params]  # TODO: it seems that all args specifiied to be 'req'. Why?

req_preprocess_args.extend(["y_col_name", "model_outdir"])  # TODO: Does 'req' mean no defaults are specified?


def run(params):
    directory = 'data/'
    dataframes = {}  # Initialize an empty dictionary to store the dataframes

    # Load the responses file into a DataFrame
    for filename in params.get('y_data_files', []):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Add to dataframe
        variable_name = filename.split('.')[0]
        dataframes[variable_name] = df

    # Process cancer files
    for filename in params.get('x_data_canc_files', []):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Select and apply the scaler for genetic features
        genetic_scaler = get_scaler(params['genetic_feature_scaling'][0])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = genetic_scaler.fit_transform(df[numeric_columns])

        # Add to dataframe
        variable_name = filename.split('.')[0]
        dataframes[variable_name] = df

    # Process drug files
    for filename in params.get('x_data_drug_files', []):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Select and apply the scaler for drug features
        drug_scaling_strategy = params['drug_feature_scaling'][0] if len(params['drug_feature_scaling']) == 1 else 'std'
        drug_scaler = get_scaler(drug_scaling_strategy)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = drug_scaler.fit_transform(df[numeric_columns])

        # Add to dataframe
        variable_name = filename.split('.')[0]
        dataframes[variable_name] = df

    # Example usage: print the first few lines of each dataframe
    for name, df in dataframes.items():
        print(f"{name} DataFrame head:")
        print(df.head())

    # Merge using common ID fields
    common_id_fields = ['CancID', 'DrugID']  # These are the columns you expect to merge on
    merged_df = merge_dataframes_on_common_id(dataframes, common_id_fields)
    print(merged_df.head())

    # Extract fields without ids and etc...
    gene_columns = merged_df.filter(regex='^ge')
    ecfp_columns = merged_df.filter(regex='^ecfp2')  # Assuming 'ecfp2_' is the correct pattern for ecfp data columns
    mordred_columns = merged_df.filter(regex='^mordred')  # Assuming 'mordred_' is the correct pattern for mordred data columns

    # Extract the target variable 'AUC'
    target = merged_df[['AUC']]  # Make sure 'AUC' is the column you want as the target

    # Concatenate all filtered columns and the target into a final DataFrame
    final_df = pd.concat([gene_columns, ecfp_columns, mordred_columns, target], axis=1)
    print(final_df.head())


    # Write the data

    # Define the output directory
    outdir = params.get('ml_data_outdir')

    # Ensure the output directory exists
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Define the path for the output CSV file
    output_csv_path = os.path.join(outdir, 'processed_data.csv')

    # Write the DataFrame to a CSV file
    final_df.to_csv(output_csv_path, index=False)
    print(f"The final processed data has been saved to: {output_csv_path}")


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
