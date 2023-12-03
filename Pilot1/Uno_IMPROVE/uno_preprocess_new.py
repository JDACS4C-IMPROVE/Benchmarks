"""Functionality for IMPROVE data handling."""

import pandas as pd

import os

from IMPROVE.improve.dataloader import get_common_samples, scale_df

from sklearn.preprocessing import (
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)


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


def merge_file_list(file_list, directory, merge_on):
    merged_df = pd.DataFrame()
    for filename in file_list:
        filepath = os.path.join(directory, filename)
        df = pd.read_parquet(filepath)
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=merge_on, how="inner")
    return merged_df


# Set IMPROVE_DATA_DIR
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception(
        "ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n"
    )
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]


directory = os.getenv(CANDLE_DATA_DIR)
x_data_canc_files = ["ge.parquet"]
x_data_drug_files = ["mordred.parquet", "ecfp2.parquet"]
y_data_files = ["rsp_full.parquet"]

gene_df = merge_file_list(x_data_canc_files, directory, "CancID")
# gene_df = gene_df[["CancID", "ge_A1BG", "ge_A1CF"]]
print(gene_df.head())
drug_df = merge_file_list(x_data_drug_files, directory, "DrugID")
# drug_df = drug_df[["DrugID", "mordred_ABC", "mordred_ABCGG"]]
print(drug_df.head())
response_df = merge_file_list(y_data_files, directory, ["CancID", "DrugID"])
response_df = response_df[["CancID", "DrugID", "AUC"]]
print(response_df.head())

# Scale Data
gene_df, _ = scale_df(gene_df, scaler_name="std")
drug_df, _ = scale_df(drug_df, scaler_name="std")

# Ensure Common Samples
gene_df, response_df = get_common_samples(gene_df, response_df, "CancID")
drug_df, response_df = get_common_samples(drug_df, response_df, "DrugID")

# Convert dataframes to multilevel
gene_df = convert_to_multilevel(gene_df, ["CancID"], "gene_info")
drug_df = convert_to_multilevel(drug_df, ["DrugID"], "drug_info")
response_df = convert_to_multilevel(
    response_df, ["CancID", "DrugID"], "response_values"
)

# Merge multilevel dataframes
processed_df = merge_multilevel_dataframe(gene_df, drug_df, response_df)
print(processed_df.head())

# Save file (hard-coded)
filename = "new_processed_data.csv"
filepath = os.path.join(directory, filename)
processed_df.to_csv(filepath, index=False)
print(f"DataFrame saved to {filepath}")
