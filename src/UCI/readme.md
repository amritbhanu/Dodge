http://archive.ics.uci.edu/ml/datasets.html

- adult - convert categorical to integer, 2 classification, big dataset
    - convert categorical to cat codes, drop missing values (final instances== 45222), binary classification (0 if <=50K else 1). Predict 1 (33% data instances)

- waveform - preprocess to only use 2 out of 3 classes.
    - used real data without noise and dropped class label 2, no missing values. Total instances=3304, (50-50% data instances, 0 and 1 label), binary classification

- Statlog (Shuttle) - , preprocess to only use 2 classes.
    - dropped all classes except 1, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=54,489 (20% relevant class), binary classification

- Statlog (Landsat Satellite) - use only 2 classes
    - dropped all classes except 1, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=2150 (30% relevant class), binary classification

- penbased http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    - dropped all classes except 2, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=2300 (50% relevant class), binary classification

- optdigits - http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    - dropped all classes except 1, 3, no missing values. Now 3 is class label 1, and 1 is 0. Total instances=1143 (50% relevant class), binary classification

- covertype - http://archive.ics.uci.edu/ml/datasets/Covertype
    - dropped all classes except 5, 4, no missing values. Now 4 is class label 1, and 5 is 0. Total instances=12,240 (34% relevant class), binary classification

- breast cancer - http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    - no missing values. Converted labels M, B to 1,0. Total instances=569 (37% relevant class), binary classification

- diabetic - http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
    - no missing values. 1 is diabetic, 0 non-diabetic. Total instances=1151 (53% relevant), binary classification


### New datasets
- annealing - https://archive.ics.uci.edu/ml/datasets/Annealing
   - dropped 26 columns as missing attributes, dropped all rows which are not 3,2. 3 is 0 label, 2 is 1 label. Converted categorical to cat.codes. 20% data distribution.
  
- autism - https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult
   - Converted categorical to cat.codes. Dropped missing value columns.

- audit - https://archive.ics.uci.edu/ml/datasets/Audit+Data
   - drop location_id
   
- bank - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
   - drop missing rows, convert categorical.
   
- blood-transfusion - https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
   - no preprocessing
   
- car - https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
   - convert categorical, convert unacc to 0 label all remaining 3 classes to 1.
   
- cardiotocography - https://archive.ics.uci.edu/ml/datasets/Cardiotocography
  - convert 1 to 0 class and 2,3 to 1 class
  
- cervical-cancer - https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
  - drop missing values row
  
- climate-sim - https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes
   - drop 1st 2 columns.
   
- contraceptive - https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
   - 1 class to 0, others to 1.
   
- credit-approval - https://archive.ics.uci.edu/ml/datasets/Credit+Approval
    - dropped missing values rows, convert categorical values.
    
- crowdsource - https://archive.ics.uci.edu/ml/datasets/Crowdsourced+Mapping
   - dropped all class labels except grass and farm. Farm is 0 label and grass is 1 label
   
- sensorless-drive - https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis
   - dropped all class labels except 1,2. 1 is label 0 and 2 is label 1.
   
- credit-default - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
   - no preprocess required.
   
- drug-consumption - https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
  - convert categorical, CL0 as 0 class, and all remaining classes as 1 class
  
- electric-stable - https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+
   - dropped stab column, converted unstable to 0 class and stable to class 1.
   
- htru2 - https://archive.ics.uci.edu/ml/datasets/HTRU2
  - no preprocess required
  
- hepmass - https://archive.ics.uci.edu/ml/datasets/HEPMASS
   - only considered 1000 weight in test file. Only used 2000 random samples (out of 3.5mil samples).
   
- liver - https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29
   - convert categorical. Label 1 as class 1 and label 2 as class 0.
   
- image - https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
   - dropped all classes except window and path. Window is label 0 and path is label 1
   
- kddcup - https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data
  - convert categorical, used only 10percent data source. Dropped all classes except normal, back dos attack. Select random 1k samples of normal out of 97k samples, used the same number of back dos labels.
  - normal. is labeled 0 and back dos is labeled 1.
  
- gamma - https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
  - converted g to label 0 and h to label 1
  
- hand - https://archive.ics.uci.edu/ml/datasets/MoCap+Hand+Postures
   - dropped few missing columns, then dropped missing value rows. Dropped all classes except 5,1. 5 is now label 0 and 1 is label 1.
   
- mushroom - https://archive.ics.uci.edu/ml/datasets/Mushroom
   - dropped missing values rows, convert categorical. Edible class is now 0 and poisonous is 1.
   
- shop-intention - https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
  - convert categorical. False is label 0 and True is label 1
  
- phishing - https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
  - convert -1 label into 0 and 1 into 1
  
- bankrupt - https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
   - used 4year.arff. Dropped missing values rows.
   
- biodegrade - https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation
  - convert NRB to label 0 and RB to 1.