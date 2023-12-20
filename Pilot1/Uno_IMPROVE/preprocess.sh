#!/bin/bash
  
#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################

# arg 1 IMPROVE_DATA_DIR
# arg 2 CANDLE_CONFIG

### Path and Name to your CANDLEized model's main Python script###

# e.g. CANDLE_MODEL=graphdrp_preprocess.py
CANDLE_MODEL=uno_preprocess_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

# Combine path and name and check if executable exists
CANDLE_MODEL=${IMPROVE_MODEL_DIR}/${CANDLE_MODEL}

if [ ! -f ${CANDLE_MODEL} ] ; then
	echo No such file ${CANDLE_MODEL}
	exit 404
fi


if [ $# -ge 2 ] ; then
        CANDLE_DATA_DIR=$1 ; shift
        FILE_OR_OPTION=$1 ; shift

        if [ -f $CANDLE_DATA_DIR/${FILE_OR_OPTION} ] ; then
		echo "$FILE_OR_OPTION is a file"
                CANDLE_CONFIG=$FILE_OR_OPTION
                CMD="python ${CANDLE_MODEL} --config_file $CANDLE_DATA_DIR/$CANDLE_CONFIG $@"
                echo "CMD = $CMD $@"

        # else passthrough $@
        else
		echo "$FILE_OR_OPTION is not a file"
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"

        fi

        #CMD="python ${CANDLE_MODEL} --config_file ${CANDLE_DATA_DIR}/${CONFIG_FILE}"
        echo "CMD = $CMD"
else
        echo "Usage: preprocess.sh DATA_DIR [CONFIG_FILE|OPTION]"
        echo "Require at least two arguments: DATA_DIR and CONFIG_FILE or command line options"
fi


# Display runtime arguments
echo "using IMPROVE_DATA_DIR ${IMPROVE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

# Set up environmental variables and execute model
# source /opt/conda/bin/activate /usr/local/conda_envs/Paccmann_MCA
IMPROVE_DATA_DIR=${IMPROVE_DATA_DIR} $CMD
