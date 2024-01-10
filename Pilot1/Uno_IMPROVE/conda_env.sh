#!/bin/bash --login

set -e

# conda create -n lgbm_py37 python=3.7 pip --yes

# IMPROVE and CANDLE
pip install git+https://github.com/JDACS4C-IMPROVE/IMPROVE@develop 
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

# Other
conda install -c conda-forge lightgbm=3.1.1 --yes # LigthGBM
pip install pandas
