#!/bin/bash --login

set -e

# conda create -n Uno_Improve python=3.7.16 pip --yes

# IMPROVE and CANDLE
# pip install git+https://github.com/JDACS4C-IMPROVE/IMPROVE@develop  # Not yet pip installable (use export to add to PYTHONPATH)
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

# Other
pip install tensorflow=2.11.0
