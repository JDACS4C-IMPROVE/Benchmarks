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


def scale_df(df, scaler_name: str="std", scaler=None, verbose: bool=False):
    """ Returns a dataframe with scaled data.

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

    if scaler is None: # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler ({scaler_name}) is not implemented (no df scaling).")
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm
    return df, scaler
# ------------------------------------------------------------


def gene_selection(df, genes_fpath, canc_col_name):
    """ Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
    # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
    # genes = list(set(genes).intersection(set(df.columns[1:])))
    genes = drp.common_elements(genes, df.columns[1:])
    cols = [canc_col_name] + genes
    return df[cols]


def compose_data_arrays(df_response: pd.DataFrame,
                        df_drug: pd.DataFrame,
                        df_cell: pd.DataFrame,
                        drug_col_name: str,
                        canc_col_name: str):
    """ Returns drug and cancer feature data, and response values.

    :params: pd.Dataframe df_response: drug response dataframe. This
             already has been filtered to three columns: drug_id,
             cell_id and drug_response.
    :params: pd.Dataframe df_drug: drug features dataframe.
    :params: pd.Dataframe df_cell: cell features dataframe.
    :params: str drug_col_name: Column name that contains the drug ids.
    :params: str canc_col_name: Column name that contains the cancer sample ids.

    :return: Numpy arrays with drug features, cell features and responses
            xd, xc, y
    :rtype: np.array
    """
    xd = [] # To collect drug features
    xc = [] # To collect cell features
    y = []  # To collect responses

    # To collect missing or corrupted data
    nan_rsp_list = []
    miss_cell = []
    miss_drug = []
    # count_nan_rsp = 0
    # count_miss_cell = 0
    # count_miss_drug = 0

    # Convert to indices for rapid lookup (??)
    df_drug = df_drug.set_index([drug_col_name])
    df_cell = df_cell.set_index([canc_col_name])

    for i in range(df_response.shape[0]):  # tuples of (drug name, cell id, response)
        if i > 0 and (i%15000 == 0):
            print(i)
        drug, cell, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            # count_nan_rsp += 1
            nan_rsp_list.append(i)
        # If drug and cell features are available
        try: # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError: # drug not found
            miss_drug.append(drug)
            # count_miss_drug += 1
        else: # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError: # cell not found
                miss_cell.append(cell)
                # count_miss_cell += 1
            else: # Both drug and cell were found
                xd.append(drug_features.values) # xd contains list of drug feature vectors
                xc.append(cell_features.values) # xc contains list of cell feature vectors
                y.append(rsp)

    # print("Number of NaN responses:   ", len(nan_rsp_list))
    # print("Number of drugs not found: ", len(miss_cell))
    # print("Number of cells not found: ", len(miss_drug))

    # # Reset index
    # df_drug = df_drug.reset_index()
    # df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)



def run(params):
    """ Execute data pre-processing for UNO model.

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
    print("\nLoading omics data ...")
    oo = drp.OmicsLoader(params)
    # print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get only gene expression dataframe
    # ---------------------

    # ------------------------------------------------------
    # [UNO] Prep omics data
    # ------------------------------------------------------
    # Gene selection (LINCS landmark genes) 
    if params["use_lincs"]:
        genes_fpath = filepath/"csa_data/raw_data/x_data/landmark_genes"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # ------------------------------------------------------
    # [Req] Load drug data
    # --------------------
    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    # print(dd)
    mod = dd.dfs['drug_mordred.tsv']  
    # -------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # [Req] All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load response
    # data, filtered by the split ids from the split files.
    # -------------------------------------------
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None

    for stage, split_file in stages.items():
        # ------------------------
        # [Req] Load response data
        # ------------------------
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        # print(rr)
        df_response = rr.dfs["response.tsv"]
        # ------------------------
        # Retain (canc, drug) response samples for which omics data is available
        ydf, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                              ref_col=params["canc_col_name"])
        print(ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        # Scale features using training data
        if stage == "train":
            # Scale data
            df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
            # Store scaler object
            if params["scaling"] is not None and params["scaling"] != "none":
                scaler_fpath = Path(params["ml_data_outdir"]) / params["scaler_fname"]
                joblib.dump(scaler, scaler_fpath)
                print("Scaler object created and stored in: ", scaler_fpath)
        else:
            # Use passed scikit scaler object
            df_canc, _ = scale_df(df_canc, scaler=scaler)

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        ydf = ydf[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(ydf, mod, df_canc, params["drug_col_name"], params["canc_col_name"])
        # print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)
        # ------------------------

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # import ipdb; ipdb.set_trace()
        # [Req] Create data name
        # data_fname = frm.build_ml_data_name(params, stage,
        #                                     file_format=params["data_format"])
        data_fname = frm.build_ml_data_name(params, stage)

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf, params, stage)

    return params["ml_data_outdir"]


def main():
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model2.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished UNO pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
