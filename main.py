
import numpy as np
import pandas as pd
from preprocessing.preprocessing import preprocessing, extract_features

# Load data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_challenge.csv')

# Preprocessing
# ------------------------------------------------
df_train_cln = preprocessing(df_train)
df_test_cln = preprocessing(df_test)

# Modelling
# ------------------------------------------------

# extract features
features = extract_features(df_train_cln)

# 
