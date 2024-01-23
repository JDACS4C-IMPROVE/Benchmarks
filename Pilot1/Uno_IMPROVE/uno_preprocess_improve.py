""" Preprocessing of raw data to generate datasets for UNO Model. """
import time
preprocess_start_time = time.time()
import sys
import os
from pathlib import Path
from typing import Dict, List, Union

# Dependencies: pandas, numpy, joblib, scikit-learn

import numpy as np
import pandas as pd
import joblib
import textwrap

from sklearn.preprocessing import (
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)

filepath = Path(__file__).resolve().parent  # [Req]

# IMPROVE imports
from improve import framework as frm

# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
from improve import (
    drug_resp_pred as drp,
)  # some funcs from dataloader.py were copied to drp


# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
app_preproc_params = [
    {
        "name": "y_data_files",  # default
        "type": str,
        "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {
        "name": "x_data_canc_files",  # required
        "type": str,
        "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {
        "name": "x_data_drug_files",  # required
        "type": str,
        "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {
        "name": "canc_col_name",
        "default": "improve_sample_id",  # default
        "type": str,
        "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {
        "name": "drug_col_name",  # default
        "default": "improve_chem_id",
        "type": str,
        "help": "Column name in the y (response) data file that contains the drug ids.",
    },
]

# 2. Model-specific params (Model: Uno)
model_preproc_params = [
    {
        "name": "use_lincs",
        "type": frm.str2bool,
        "default": False,
        "help": "Flag to indicate if using landmark genes.",
    },
    {
        "name": "gene_scaling",
        "type": str,
        "default": "std",
        "choice": ["std", "minmax", "miabs", "robust"],
        "help": "Scaler for gene expression data.",
    },
    {
        "name": "ge_scaler_fname",
        "type": str,
        "default": "x_data_gene_expression_scaler.gz",
        "help": "File name to save the gene expression scaler object.",
    },
    {
        "name": "drug_scaling",
        "type": str,
        "default": "std",
        "choice": ["std", "minmax", "miabs", "robust"],
        "help": "Scaler for gene expression data.",
    },
    {
        "name": "md_scaler_fname",
        "type": str,
        "default": "x_data_mordred_scaler.gz",
        "help": "File name to save the Mordred scaler object.",
    },
    {
        "name": "preprocess_debug",
        "type": bool,
        "default": True,
        "help": "Debug mode to show data",
    },
]

preprocess_params = app_preproc_params + model_preproc_params


# ------------------------------------------------------------


def print_duration(activity, start_time, end_time):
    """
    activity (str): Description of the activity.
    duration (int): Duration in minutes.
    """
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")


def gene_selection(df: pd.DataFrame, genes_fpath: Union[Path, str], canc_col_name: str):
    """Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
    # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
    genes = list(set(genes).intersection(set(df.columns[1:])))
    # genes = drp.common_elements(genes, df.columns[1:])
    cols = [canc_col_name] + genes
    return df[cols]


def scale_df(
    df: pd.DataFrame, scaler_name: str = "std", scaler=None, verbose: bool = False
):
    """Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        df: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = df.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(
                f"The specified scaler ({scaler_name}) is not implemented (no df scaling)."
            )
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm
    return df, scaler


def compose_data_arrays(
    df_cell: pd.DataFrame,
    df_drug: pd.DataFrame,
    df_response: pd.DataFrame,
    canc_col_name: str,
    drug_col_name: str,
):
    """Returns drug and cancer feature data, and response values.

    Args:
        df_response (pd.Dataframe): drug response df. This already has been
            filtered to three columns: drug_id, cell_id and drug_response.
        df_drug (pd.Dataframe): drug features df.
        df_cell (pd.Dataframe): cell features df.
        drug_col_name (str): Column name that contains the drug ids.
        canc_col_name (str): Column name that contains the cancer sample ids.

    Returns:
        np.array: arrays with drug features, cell features and responses
            xd, xc, y

    Justification:
        According to some searching, appending to a list and then converting to a
        dataframe is faster than appending to a datafrane for very large sets
    """
    xc = []  # To collect cell features
    xd = []  # To collect drug features
    y = []  # To collect responses

    # To collect missing or corrupted data
    miss_cell = []
    miss_drug = []
    nan_rsp_list = []
    count_miss_cell = 0
    count_miss_drug = 0
    count_nan_rsp = 0

    df_cell = df_cell.set_index([canc_col_name])
    df_drug = df_drug.set_index([drug_col_name])

    for i in range(df_response.shape[0]):  # tuples of (cell id, drug name, response)
        if i > 0 and (i % 15000 == 0):
            print(i)
        cell, drug, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            count_nan_rsp += 1
            nan_rsp_list.append(i)
        # If drug and cell features are available
        try:  # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError:  # drug not found
            miss_drug.append(drug)
            count_miss_drug += 1
        else:  # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError:  # cell not found
                miss_cell.append(cell)
                count_miss_cell += 1
            else:  # Both drug and cell were found
                xd.append(
                    drug_features.values
                )  # xd contains list of drug feature vectors
                xc.append(
                    cell_features.values
                )  # xc contains list of cell feature vectors
                y.append(rsp)

    print("Number of NaN responses:   ", len(nan_rsp_list))
    print("Number of drugs not found: ", len(miss_cell))
    print("Number of cells not found: ", len(miss_drug))

    return pd.DataFrame(xc), pd.DataFrame(xd), pd.DataFrame(y)


def run(params: Dict):
    """Execute data pre-processing for UNO model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    # Create output dirto save preprocessed data)
    frm.create_outdir(outdir=params["ml_data_outdir"])
    # ------------------------------------------------------

    temp_start_time = time.time()
    # ------------------------------------------------------
    # [Req] Load omics data
    # ------------------------------------------------------
    print("\nLoading omics data.")
    omics_obj = drp.OmicsLoader(params)
    gene_expression_file = params["x_data_canc_files"][0][0]
    print("Loading file: ", gene_expression_file)
    ge = omics_obj.dfs[gene_expression_file]  # return gene expression
    ge["improve_sample_id"] = ge['improve_sample_id'].astype(str)  # To fix mixed dytpes error
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Load drug data
    # ------------------------------------------------------
    print("\nLoading drugs data.")
    drugs_obj = drp.DrugsLoader(params)
    drug_data_file = params["x_data_drug_files"][0][0]
    print("Loading file: ", drug_data_file)
    md = drugs_obj.dfs[drug_data_file]  # return mordred drug descriptors
    md = md.reset_index()  # Needed to do scaling and merging as wanted
    md["improve_chem_id"] = md['improve_chem_id'].astype(str)  # To fix mixed dytpes error
    # ------------------------------------------------------
    temp_end_time = time.time()
    print("")
    print_duration("Loading Data", temp_start_time, temp_end_time)

    # Check loaded data if debug mode on
    if params["preprocess_debug"]:
        print("Loaded Gene Expression:")
        print(ge.head())
        print(ge.shape)
        print("")
        print("Loaded Mordred Descriptors:")
        print(md.head())
        print(md.shape)
        print("")

    # ------------------------------------------------------
    # To-Do: USE LINCS
    # ------------------------------------------------------
    # Gene selection (LINCS landmark genes)
    # if params["use_lincs"]:
    #     genes_fpath = filepath + "/raw_data/x_data/landmark_genes"
    #     ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])
    # ------------------------------------------------------

    temp_start_time = time.time()
    # ------------------------------------------------------
    # Data prep to create scaler on
    # ------------------------------------------------------
    # Load and combine train and val responses
    rsp_tr = drp.DrugResponseLoader(
        params, split_file=params["train_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(
        params, split_file=params["val_split_file"], verbose=False
    ).dfs["response.tsv"] # Note: Study column is integer but expects str giving warning
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)
    # Only keep relevant parts of response
    rsp = rsp[[params["canc_col_name"], params["drug_col_name"], 'auc']]

    # Keep only common samples between response, drug, and cancer information
    # Response
    rsp_sub = rsp.merge(
       ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner"
    )
    rsp_sub = rsp_sub.merge(
       md[params["drug_col_name"]], on=params["drug_col_name"], how="inner"
    )
    # Gene expression
    ge_sub = ge[
        ge[params["canc_col_name"]].isin(rsp_sub[params["canc_col_name"]])
    ].reset_index(drop=True)
    # Mordred descriptors
    md_sub = md[
        md[params["drug_col_name"]].isin(rsp_sub[params["drug_col_name"]])
    ].reset_index(drop=True)

    # Check shapes if debug mode on
    if params["preprocess_debug"]:
        print(textwrap.dedent(f"""
            Gene Expression Shape Before Subsetting With Response: {ge.shape}
            Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
            Mordred Shape Before MSubsetting With Response: {md.shape}
            Mordred Shape After Subsetting With Response: {md_sub.shape}
            Response Shape Before Merging With Data: {rsp.shape}
            Response Shape After Merging With Data: {rsp_sub.shape}
        """))


    # ------------------------------------------------------
    # Create Feature Scaler
    # ------------------------------------------------------
    # Note: these scale before merging, so doesn't overweight according
    # to common treatments/cell lines
    print("\nCreating Feature Scalers\n")

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["ge_scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # Scale Mordred descriptors
    _, md_scaler = scale_df(md_sub, scaler_name=params["md_scaling"])
    md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub   # Clean Up
    # ------------------------------------------------------
    temp_end_time = time.time()
    print_duration("Creating Scaler", temp_start_time, temp_end_time)


    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():

        split_start_time = time.time()
        print(f"Stage: {stage.upper()}")
        # --------------------------------------------------
        # Data prep
        # --------------------------------------------------
        # Load split responses
        rsp = drp.DrugResponseLoader(params, split_file=split_file, verbose=False).dfs[
            "response.tsv"
        ]
        # Only keep relevant parts of response
        rsp = rsp[[params["canc_col_name"], params["drug_col_name"], 'auc']]

        # Keep only common samples between response, drug, and cancer information
        # Response
        rsp_sub = rsp.merge(
            ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner"
        )
        rsp_sub = rsp_sub.merge(
            md[params["drug_col_name"]], on=params["drug_col_name"], how="inner"
        )
        # Gene Expression
        ge_sub = ge[
            ge[params["canc_col_name"]].isin(rsp_sub[params["canc_col_name"]])
        ].reset_index(drop=True)
        # Mordred Descriptors
        md_sub = md[
            md[params["drug_col_name"]].isin(rsp_sub[params["drug_col_name"]])
        ].reset_index(drop=True)

        # Check shapes if debug mode on
        if params["preprocess_debug"]:
            print(textwrap.dedent(f"""
                Gene Expression Shape Before Subsetting With Response: {ge.shape}
                Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
                Mordred Shape Before Subsetting With Response: {md.shape}
                Mordred Shape After Subsetting With Response: {md_sub.shape}
                Response Shape Before Merging With Data: {rsp.shape}
                Response Shape After Merging With Data: {rsp_sub.shape}
            """))

        # Scale features
        temp_start_time = time.time()
        print("\nScaling data")
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # scale gene expression
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)  # scale Mordred descriptors
        temp_end_time = time.time()
        print_duration(f"Applying Scaler to {stage.capitalize()}", temp_start_time, temp_end_time)
        if params["preprocess_debug"]:
            print("Gene Expression Scaled:")
            print(ge_sc.head())
            print(ge_sc.shape)
            print("")
            print("Mordred Descriptors Scaled:")
            print(md_sc.head())
            print(md_sc.shape)
            print("")

        # --------------------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # --------------------------------
        # [Req] Build data name
        data_fname = frm.build_ml_data_name(params, stage)

        # TO-DO: Standardize the saving process

        # Compose data
        temp_start_time = time.time()
        print("Composing Data")
        xc, xd, y = compose_data_arrays(
            ge_sc, md_sc, rsp_sub, params["canc_col_name"], params["drug_col_name"]
        )
        temp_end_time = time.time()
        print_duration(f"Composing {stage.capitalize()} Dataframes", temp_start_time, temp_end_time)
        print(stage.capitalize(), "data --> xc ", xc.shape, "xd ", xd.shape, "y ", y.shape, "\n")

        # Show dataframes if on debug mode
        if params["preprocess_debug"]:
            print("Gene Expression Dataframe:")
            print(xc.head())
            print(xc.shape)
            print("")
            print("Mordred Descriptors Dataframe:")
            print(xd.head())
            print(xd.shape)
            print("")
            print("Responses Dataframe:")
            print(y.head())
            print(y.shape)
            print("")

        # Construct file paths
        xc_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_x_canc.parquet"
        xd_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_x_drug.parquet"
        y_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_y_data.parquet"

        # Save dataframes to the constructed file paths
        temp_start_time = time.time()
        print("Saving Dataframes to Parquet")
        xc.columns = xc.columns.map(str)
        xd.columns = xd.columns.map(str)
        y.columns = y.columns.map(str)
        print("Saving Gene Expression")
        xc.to_parquet(xc_fpath, index=False)
        print("Saving Mordred Descriptors")
        xd.to_parquet(xd_fpath, index=False)
        print("Saving Responses")
        y.to_parquet(y_fpath, index=False)
        temp_end_time = time.time()
        print_duration(f"Saving {stage.capitalize()} Dataframes", temp_start_time, temp_end_time)

        split_end_time = time.time()
        print_duration(f"Processing {stage.capitalize()} Data", split_start_time, split_end_time)

    # Print total duration
    preprocess_end_time = time.time()
    print_duration(
        f"Preprocessing Data (All)", preprocess_start_time, preprocess_end_time
    )

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # [Req]
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        additional_definitions=preprocess_params,
        # required=req_preprocess_params,
        required=None,
    )
    ml_data_outdir = run(params)
    print(
        "\nFinished UNO pre-processing (transformed raw DRP data to model input ML data)."
    )


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
