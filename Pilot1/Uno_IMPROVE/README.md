# Files in the repo
- Modified for IMPROVE (pick one configuration of drug):
   - uno_preprocess.py
   - uno_train.py
   - uno_infer.py
   - EXTRA: uno_baseline_keras.py - keep this file as a master file that calls the above three files
   - uno_default_model.txt

# Dependencies
- Install notes: PyArrow 12.0.1

In order to run the modified version of the code, you need to run the following commands:
```
export IMPROVE_DATA_DIR=<UNO_PATH>csa_data or other as per desired directory
export PYTHONPATH=<IMPROVE_LIBRARY>/:$PYTHONPATH 
python uno_preprocess.py
python uno_train.py
python uno_infer.py
```
