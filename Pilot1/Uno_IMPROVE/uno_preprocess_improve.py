""" Preprocessing of raw data to generate datasets for UNO Model. """
import sys
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import joblib

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

# filepath = Path(__file__).resolve().parent
filepath = os.getenv("IMPROVE_DATA_DIR")

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


model_preproc_params = [
    {
        "name": "use_lincs",
        "type": frm.str2bool,
        "default": True,
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
]

preprocess_params = app_preproc_params + model_preproc_params


# ------------------------------------------------------------


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


def compose_data_arrays(
    df_response: pd.DataFrame,
    df_drug: pd.DataFrame,
    df_cell: pd.DataFrame,
    drug_col_name: str,
    canc_col_name: str,
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
    """
    xd = []  # To collect drug features
    xc = []  # To collect cell features
    y = []  # To collect responses

    # To collect missing or corrupted data
    nan_rsp_list = []
    miss_cell = []
    miss_drug = []
    # count_nan_rsp = 0
    # count_miss_cell = 0
    # count_miss_drug = 0

    df_drug = df_drug.set_index([drug_col_name])
    df_cell = df_cell.set_index([canc_col_name])

    for i in range(df_response.shape[0]):  # tuples of (drug name, cell id, response)
        if i > 0 and (i % 15000 == 0):
            print(i)
        drug, cell, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            # count_nan_rsp += 1
            nan_rsp_list.append(i)
        # If drug and cell features are available
        try:  # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError:  # drug not found
            miss_drug.append(drug)
            # count_miss_drug += 1
        else:  # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError:  # cell not found
                miss_cell.append(cell)
                # count_miss_cell += 1
            else:  # Both drug and cell were found
                xd.append(
                    drug_features.values
                )  # xd contains list of drug feature vectors
                xc.append(
                    cell_features.values
                )  # xc contains list of cell feature vectors
                y.append(rsp)

    # print("Number of NaN responses:   ", len(nan_rsp_list))
    # print("Number of drugs not found: ", len(miss_cell))
    # print("Number of cells not found: ", len(miss_drug))

    # # Reset index
    # df_drug = df_drug.reset_index()
    # df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)


def run(params: Dict):
    """Execute data pre-processing for UNO model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # import pdb; pdb.set_trace()

    # --------------------------------------------
    # Check consistency of parameter specification
    # --------------------------------------------
    # check_parameter_consistency(params)

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ----------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    # Create outdir for ML data (to save preprocessed data)
    # processed_outdir = frm.create_ml_data_outdir(params)  # creates params["ml_data_outdir"]
    frm.create_outdir(outdir=params["ml_data_outdir"])
    # ----------------------------------------

    # ------------------------------------------------------
    # Construct data frames for drug and cell features
    # ------------------------------------------------------
    # df_drug, df_cell_all, smile_graphs = build_common_data(params, indtd)

    # ------------------------------------------------------
    # [Req] Load omics data
    # ---------------------
    print("\nLoads omics data.")
    omics_obj = drp.OmicsLoader(params)
    # print(omics_obj)
    gene_expression_file = params["x_data_canc_files"][0][0]
    print("Loading file: ", gene_expression_file)
    ge = omics_obj.dfs[gene_expression_file]  # return gene expression
    # print(ge.head())
    # print(ge.shape)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Load drug data
    # ------------------------------------------------------
    print("\nLoad drugs data.")
    drugs_obj = drp.DrugsLoader(params)
    # print(drugs_obj)
    drug_data_file = params["x_data_drug_files"][0][0]
    print("Loading file: ", drug_data_file)
    md = drugs_obj.dfs[drug_data_file]  # return mordred drug descriptors
    md = md.reset_index()  # Needed to do scaling and merging as wanted
    # ------------------------------------------------------

    # ------------------------------------------------------
    # To-Do: USE LINCS
    # ------------------------------------------------------
    # Gene selection (LINCS landmark genes)
    # if params["use_lincs"]:
    #     genes_fpath = filepath + "/raw_data/x_data/landmark_genes"
    #     ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    print("Create feature scaler.")
    rsp_tr = drp.DrugResponseLoader(
        params, split_file=params["train_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(
        params, split_file=params["val_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge(
        ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner"
    )
    rsp = rsp.merge(
        md[params["drug_col_name"]], on=params["drug_col_name"], how="inner"
    )
    ge_sub = ge[
        ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])
    ].reset_index(drop=True)
    md_sub = md[
        md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])
    ].reset_index(drop=True)

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

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():
        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params, split_file=split_file, verbose=False).dfs[
            "response.tsv"
        ]

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge(
            ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner"
        )
        rsp = rsp.merge(
            md[params["drug_col_name"]], on=params["drug_col_name"], how="inner"
        )
        ge_sub = ge[
            ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])
        ].reset_index(drop=True)
        md_sub = md[
            md[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])
        ].reset_index(drop=True)

        # Scale features
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # scale gene expression
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)  # scale Mordred descriptors

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        rsp = rsp[
            [params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]
        ]

        # print(ge.head())
        # print(mod.head())
        # print(ydf.head())

        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(
            rsp, md_sc, ge_sc, params["drug_col_name"], params["canc_col_name"]
        )
        print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # Make numpy arrays dataframes
        xc_df = pd.DataFrame(xc)
        xd_df = pd.DataFrame(xd)
        y_df = pd.DataFrame(y)

        # TestBedDataset???
        # data_fname = frm.build_ml_data_name(params, stage)

        # Construct file paths
        xc_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_x_canc.csv"
        xd_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_x_drug.csv"
        y_fpath = Path(params[f"{stage}_ml_data_dir"]) / f"{stage}_y_data.csv"

        # Save dataframes to the constructed file paths
        xc_df.to_csv(xc_fpath, index=False)
        xd_df.to_csv(xd_fpath, index=False)
        y_df.to_csv(y_fpath, index=False)

        # ------------------------

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # import ipdb; ipdb.set_trace()
        # [Req] Create data name
        # data_fname = frm.build_ml_data_name(params, stage,
        #                                     file_format=params["data_format"])
        # data_fname = frm.build_ml_data_name(params, stage)

        # [Req] Save y dataframe for the current stage
        # frm.save_stage_ydf(ydf, params, stage)

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # Set IMPROVE_DATA_DIR
    if os.getenv("IMPROVE_DATA_DIR") is None:
        raise Exception(
            "ERROR ! Required system variable not specified.  \
                        You must define IMPROVE_DATA_DIR ... Exiting.\n"
        )

    filepath = os.getenv("IMPROVE_DATA_DIR")

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
