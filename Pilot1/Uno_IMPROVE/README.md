# Files in the repo
- Modified for IMPROVE (pick one configuration of drug):
   - uno_default_model.txt
   - uno_preprocess.py
   - uno_train.py
   - uno_infer.py

# Conda Run (Miniconda version: 23.11.0)
- conda create Uno_IMPROVE python=3.7.16
- conda activate Uno_IMPROVE
- conda config --add channels conda-forge
- conda install tensorflow-gpu=2.10.0
- pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
- pip install protobuf==3.20.0
- git clone https://github.com/JDACS4C-IMPROVE/IMPROVE.git
- export PYTHONPATH=<IMPROVE_LIBRARY_PATH>/:$PYTHONPATH
- pip install pyarrow==12.0.1
- pip install scikit-learn==1.0.2
- pip install joblib 1.3.2


In order to run the modified version of the code, you need to run the following commands:
```
single_drug_drp/benchmark-data-pilot1/csa_data/
export IMPROVE_DATA_DIR=<DESIRED_DATA_DIR>
wget --cut-dirs=8 -P ~/$IMPROVE_DATA_DIR -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/
export PYTHONPATH=<IMPROVE_LIBRARY>/:$PYTHONPATH 
python uno_preprocess_improve.py
python uno_train_improve.py
```