""" Preprocessing of raw data to generate datasets for UNO Model. """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# IMPROVE imports
import sys
# Append the IMPROVE directory
file_directory = os.path.dirname(os.path.abspath(__file__))
improve_directory = os.path.join(file_directory, 'IMPROVE')
sys.path.append(improve_directory)
from improve import framework as frm
# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
# from improve import drug_resp_pred as drp  # some funcs from dataloader.py were copied to drp

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

preprocess_params = model_preproc_params + drp_preproc_params

req_preprocess_args = [ll["name"] for ll in drp_preproc_params]  # TODO: it seems that all args specifiied to be 'req'. Why?

req_preprocess_args.extend(["y_col_name", "model_outdir"])  # TODO: Does 'req' mean no defaults are specified?


def run(params):

    # Check CANDLE_DATA_DIR to read data from
    if 'CANDLE_DATA_DIR' not in os.environ:
        sys.exit("Error: CANDLE_DATA_DIR environment variable is not set")
    else:
        directory = os.environ['CANDLE_DATA_DIR']


    # Load the responses file into a DataFrame
    rsp_dataframes = []
    i=0
    for filename in params.get('y_data_files', []):
        response_path="raw_data/y_data/"+str(filename[i])
        file_path = os.path.join(directory, response_path)
        df = pd.read_csv(file_path)
        # Add to dataframe
        rsp_dataframes.append(df)
        i+=1

    # Merge dataframe
    rsp_df = rsp_dataframes[0]  # start with the first dataframe
    for df in rsp_dataframes[1:]:
        rsp_df = pd.merge(rsp_df, df, on=["CancID", "DrugID"], how="inner")
    # Select the desired metric out
    print(rsp_df.columns)
    rsp_df = rsp_df[['CancID', 'DrugID', 'AUC']]
    # Do Multi-indexing of columns
    id_columns = [('ID', 'CancID'), ('ID', 'DrugID')]
    response_value_columns = [('response_values', 'AUC')]
    new_columns = pd.MultiIndex.from_tuples(id_columns + response_value_columns)
    rsp_df.columns = new_columns
    print("Response Datafile:")
    print(rsp_df.head())
    print(rsp_df.shape)


    # Process gene expression files
    ge_dataframes = []
    for filename in params.get('x_data_canc_files', []):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        # Select and apply the scaler for genetic features
        genetic_scaler = get_scaler(params['genetic_feature_scaling'][0])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = genetic_scaler.fit_transform(df[numeric_columns])
        # Add to dataframe
        ge_dataframes.append(df)

    # Merge dataframe
    ge_df = ge_dataframes[0]  # start with the first dataframe
    for df in ge_dataframes[1:]:
        ge_df = pd.merge(ge_df, df, on=["CancID"], how="inner")
    # Do Multi-indexing of columns
    id_column = [('ID', 'CancID')]
    gene_expression_columns = [('gene_expression', col) for col in ge_df.columns if col != 'CancID']
    new_columns = pd.MultiIndex.from_tuples(id_column + gene_expression_columns)
    ge_df.columns = new_columns
    print("Cancer Datafile")
    print(ge_df.head())
    print(ge_df.shape)


    # Process drug files
    drug_dataframes = []
    for filename in params.get('x_data_drug_files', []):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        # Select and apply the scaler for drug features
        drug_scaler = get_scaler(params['drug_feature_scaling'][0])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = drug_scaler.fit_transform(df[numeric_columns])
        # Add to dataframe
        drug_dataframes.append(df)

    # Merge dataframe
    drug_df = drug_dataframes[0]  # start with the first dataframe
    for df in drug_dataframes[1:]:
        drug_df = pd.merge(drug_df, df, on=["DrugID"], how="inner")
    # Do Multi-indexing of columns
    id_column = [('ID', 'DrugID')]
    drug_expression_columns = [('drug_expression', col) for col in drug_df.columns if col != 'DrugID']
    new_columns = pd.MultiIndex.from_tuples(id_column + drug_expression_columns)
    drug_df.columns = new_columns
    print("Drug Datafile")
    print(drug_df.head())
    print(drug_df.shape)


    # Make combined multilevel df
    # Make ID columns by copying from rsp_df
    combined_df = rsp_df.loc[:, pd.IndexSlice['ID', :]]
    # Combine all dfs
    combined_df = pd.merge(combined_df, ge_df, on=[('ID', 'CancID')], how='inner')
    combined_df = pd.merge(combined_df, drug_df, on=[('ID', 'DrugID')], how='inner')
    combined_df = pd.merge(combined_df, rsp_df, on=[('ID', 'CancID'), ('ID', 'DrugID')], how='inner')
    print("Combined Datafile")
    print(combined_df.head())
    print(combined_df.shape)


    # Write the data
    # Define the output directory
    outdir = params.get('ml_data_outdir')
    # Ensure the output directory exists
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # Define the path for the output CSV file
    output_csv_path = os.path.join(outdir, 'processed_data.csv')
    # Write the DataFrame to a CSV file
    combined_df.to_csv(output_csv_path, index=False)
    print(f"The final processed data has been saved to: {output_csv_path}")


def main():
    params = frm.initialize_parameters(
        file_directory,
        default_model="uno_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished UNO pre-processing (transformed raw data to model input ML data).")


if __name__ == "__main__":
    main()
