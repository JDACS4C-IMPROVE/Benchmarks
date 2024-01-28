#!/bin/bash
  
#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 IMPROVE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path and Name to your CANDLEized model's main Python script###

# e.g. CANDLE_MODEL=graphdrp_baseline_pytorch.py
CANDLE_MODEL_PRE=uno_preprocess_improve.py
CANDLE_MODEL=uno_train_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

# Combine path and name and check if executable exists
CANDLE_MODEL=${IMPROVE_MODEL_DIR}/${CANDLE_MODEL}
if [ ! -f ${CANDLE_MODEL} ] ; then
	echo No such file ${CANDLE_MODEL}
	exit 404
fi


if [ $# -lt 2 ] ; then
        echo "Illegalnumber of paramaters"
        echo "Illegal number of parameters"
        echo "CUDA_VISIBLE_DEVICES IMPROVE_DATA_DIR are required"
        exit -1
fi

if [ $# -eq 2 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        IMPROVE_DATA_DIR=$1 ; shift
        CMD_PRE="python ${CANDLE_MODEL_PRE}"
        CMD="python ${CANDLE_MODEL}"
        echo "CMD = $CMD"

elif [ $# -ge 3 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        IMPROVE_DATA_DIR=$1 ; shift

        # if $3 is a file, then set candle_config
        if [ -f $IMPROVE_DATA_DIR/$1 ] ; then
		echo "$1 is a file"
                CANDLE_CONFIG=$1 ; shift
                CMD_PRE="python ${CANDLE_MODEL_PRE} --config_file $CANDLE_CONFIG $@"
                CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
                echo "CMD = $CMD $@"

        # else passthrough $@
        else
		echo "$1 is not a file"
                CMD_PRE="python ${CANDLE_MODEL_PRE} $@"
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"

        fi
fi

# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using IMPROVE_DATA_DIR ${IMPROVE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

# Set up environmental variables and execute model
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} IMPROVE_DATA_DIR=${IMPROVE_DATA_DIR}
$CMD_PRE
echo "CHECKPOINT 1 Finished Preprocessing ---------------------------------------------"
$CMD