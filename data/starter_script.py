#!/usr/bin/python
# Copyright (c)2018 Accenture and/or its affiliates.  All Rights Reserved.  
# You may not use, copy, modify, and/or distribute this code and/or its documentation without permission from Accenture.
# Please contact the Applied Intelligence team and/or Frode Huse Gjendem (lead) with any questions.

# brief: This is the starter script for the Accenture Datathon 2018 Competition.
import os
import pandas as pd

# You can use this function to calculate AUC
# from sklearn.metrics import roc_auc_score
# roc_auc_score(y_true, y_predicted)

# -----1. Set configuration & Data Import.----- 
# Enter your input data and output data paths below.
PATH = os.getcwd() # Otherwise use your own path
OUTPATH = os.getcwd()
# Set the input data folder as default path.
os.chdir(PATH)

# ----- Data Import -----
# Read the test files.
test = pd.read_csv("test.csv")

# -----2. Model.----- 
test["MDR"] = 0

# -----3. Data Transformation.-----
submission = test.loc[:, ['ID', 'MDR']]  

# -----5. Save the submission.----- 
# Write the final CSV file.
submission.to_csv(OUTPATH+"/sample-submission.csv", encoding='utf-8', index=False)

# Please, remember than in order to make the submission you need to create a .zip file with the csv