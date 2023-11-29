"""Functionality for IMPROVE data handling."""

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

import os

from sklearn.preprocessing import (
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)


def get_common_samples(
    df1: pd.DataFrame, df2: pd.DataFrame, ref_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Search for common data in reference column and retain only .

    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2 after filtering for common data.

    Example:
        Before:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579
        CCLE	ACH-000475	Drug_490	0.213

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000179  5.202025844609336	3.5046203924035524	3.5058909297299574
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709

        After:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709
    """
    # Retain df1 and df2 samples with common ref_col
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def scale_df(df, scaler_name: str = "std", scaler=None, verbose: bool = False):
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
    return df


def convert_to_multilevel(df, id_cols, level_name):
    # Create multi-level id columns
    id_columns = df[id_cols]
    multilevel_id_columns = pd.MultiIndex.from_tuples(
        [("ID", col) for col in id_columns.columns]
    )
    id_columns.columns = multilevel_id_columns
    # Create multi-level columns for level_name
    data_columns = df.drop(columns=id_cols)
    multilevel_data_columns = pd.MultiIndex.from_tuples(
        [(level_name, col) for col in data_columns.columns]
    )
    data_columns.columns = multilevel_data_columns
    # Combine the multilevel dataframes
    df = pd.concat([id_columns, data_columns], axis=1)
    return df


def merge_multilevel_dataframe(gene_df, drug_df, response_df):
    gene_response_merged = pd.merge(
        gene_df,
        response_df,
        how="inner",
        left_on=[("ID", "CancID")],
        right_on=[("ID", "CancID")],
    )
    # Drop so no dupe
    gene_response_merged = gene_response_merged.drop(
        columns=[("response_values", "AUC")]
    )
    # print(gene_response_merged.head())
    drug_response_merged = pd.merge(
        drug_df,
        response_df,
        how="inner",
        left_on=[("ID", "DrugID")],
        right_on=[("ID", "DrugID")],
    )
    # print(drug_response_merged.head())
    merged_df = pd.merge(
        gene_response_merged,
        drug_response_merged,
        how="inner",
        left_on=[("ID", "CancID"), ("ID", "DrugID")],
        right_on=[("ID", "CancID"), ("ID", "DrugID")],
    )

    # Separate out columns
    id_columns = merged_df.loc[:, ("ID", slice(None))]
    # print(id_columns)
    gene_columns = merged_df.loc[:, ("gene_info", slice(None))]
    # print(gene_columns)
    drug_columns = merged_df.loc[:, ("drug_info", slice(None))]
    # print(drug_columns)
    response_columns = merged_df.loc[:, ("response_values", slice(None))]
    # print(response_columns)

    # Combine all the columns nicely
    merged_df = pd.concat(
        [id_columns, gene_columns, drug_columns, response_columns], axis=1
    )

    # print(merged_df.head())

    return merged_df


directory = "/mnt/c/Users/rylie/Coding/UNO/Benchmarks/Pilot1/Uno_IMPROVE/ml_data/"  # This should use CANDLE_DATA_DIR

gene_file = file_path = os.path.join(directory, "ge.csv")
gene_df = pd.read_csv(gene_file)
gene_df = scale_df(gene_df, scaler_name="std")
# print(gene_df.head())
# print(gene_df.shape)
# gene_df = gene_df[["CancID", "ge_A1BG", "ge_A1CF"]]

drug_file = file_path = os.path.join(directory, "mordred.csv")
drug_df = pd.read_csv(drug_file)
drug_df = scale_df(drug_df, scaler_name="std")
# print(drug_df.head())
# print(drug_df.shape)
# drug_df = drug_df[["DrugID", "mordred_ABC", "mordred_ABCGG"]]

response_file = file_path = os.path.join(directory, "rsp_full.csv")
response_df = pd.read_csv(response_file)
response_df = response_df[["CancID", "DrugID", "AUC"]]
# print(response_df.head())
# print(response_df.shape)

# Ensure Common Samples
gene_df, response_df = get_common_samples(gene_df, response_df, "CancID")
drug_df, response_df = get_common_samples(drug_df, response_df, "DrugID")

# Convert dataframes to multilevel
gene_df = convert_to_multilevel(gene_df, ["CancID"], "gene_info")
drug_df = convert_to_multilevel(drug_df, ["DrugID"], "drug_info")
response_df = convert_to_multilevel(
    response_df, ["CancID", "DrugID"], "response_values"
)
# print(gene_df.head())
# print(drug_df.head())
# print(response_df.head())

# Merge multilevel dataframes
processed_df = merge_multilevel_dataframe(gene_df, drug_df, response_df)
print(processed_df.head())

# Save file (hard-coded)
filename = "new_processed_data.csv"
filepath = os.path.join(directory, filename)
processed_df.to_csv(filepath, index=False)
print(f"DataFrame saved to {filepath}")
