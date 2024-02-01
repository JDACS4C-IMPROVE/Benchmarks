#!/bin/bash

# arg 1 IMPROVE_DATA_DIR

### Path and Name to your CANDLEized model's main Python script###

# Preprocess python script here requires the IMPROVE cross-study data
# e.g. for experiments CCLE-CCLE, CTRPv2-gCSI. Current list of datasets
# are: gCSI, CCLE, CTRPv2, GDSCv1, GDSCv2
CANDLE_MODEL_PRE=uno_preprocess_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
CANDLE_MODEL_DIR=${CANDLE_MODEL_DIR:-$( dirname -- "$0" )}

# Combine path and name and check if executable exists
CANDLE_MODEL_PRE=${CANDLE_MODEL_DIR}/${CANDLE_MODEL_PRE}
if [ ! -f ${CANDLE_MODEL} ] ; then
    echo "No such file ${CANDLE_MODEL}"
    exit 404
fi

# Check if IMPROVE_DATA_DIR is provided
if [ $# -lt 1 ] ; then
    echo "Illegal number of parameters"
    echo "Usage: $0 IMPROVE_DATA_DIR"
    exit 1
fi

# Assign the first argument to IMPROVE_DATA_DIR
IMPROVE_DATA_DIR=$1

# Command to execute the Python script
CMD="python ${CANDLE_MODEL_PRE} ${IMPROVE_DATA_DIR}"
echo "CMD = $CMD"

# Display runtime arguments
echo "using IMPROVE_DATA_DIR ${IMPROVE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

# Set up environmental variables and execute model
# source /opt/conda/bin/activate /usr/local/conda_envs/Paccmann_MCA

IMPROVE_DATA_DIR=${IMPROVE_DATA_DIR} $CMD