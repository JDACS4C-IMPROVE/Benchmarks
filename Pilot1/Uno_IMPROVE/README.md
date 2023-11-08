# Files in the repo:
- Original: 
  - uno_data.py - data loading and preprocessing   
  - uno_baseline_keras.py
  - uno_default_model.txt

- Modified for IMPROVE (pick one configuration of drug):
   - uno_preprocess.py
   - uno_train.py
   - uno_infer.py
   - EXTRA: uno_baseline_keras.py - keep this file as a master file that calls the above three files
   - uno_default_model.txt

In order to run the modified version of the code, you need to run the following commands:
```
python uno_preprocess.py
python uno_train.py
python uno_infer.py
```
