

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV #Perforing grid search

from preprocessing.preprocessing import preprocessing, extract_features, PCA_r, generate_rooms

from xgboost import XGBClassifier
from models.xgboost_model import xgboost_fit

from matplotlib import pyplot
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# Variables for PCA (diagnostic)
features_diagnostic = [
'dummy_others.LL',
"dummy_Cancer.linfoproliferativo",
"dummy_SMD",
"dummy_Leucemia.cronica",
"dummy_LAL",
"dummy_EICH",
"dummy_SMPC",
"dummy_Cancer.solido",
"dummy_LMC",
"dummy_TLPT",
"dummy_others.LM",
"dummy_Mieloma.like",
"dummy_LLC"]

# Variables for PCA (antibiotic)
features_antibiotic = [
'AMIKACINA_.MG.',
'AMOXICILINA_.MG.',
'AMPICILINA_.MG.',
'AZITROMICINA_VIAL_.MG.',
'AZTREONAM_.MG.',
'CEFAZOLINA_.MG.',
'CEFIXIMA_.MG.',
'CEFOTAXIMA_.MG.',
'CEFOXITINA_.MG.',
'CEFTAROLINA_FOSAMIL_.MG.',
'CEFTAZIDIMA_.MG.',
'CEFTIBUTENO_.MG.',
'CEFTOLOZANO_.UND.',
'CEFTRIAXONA_.MG.',
'CEFUROXIMA.AXETILO_.MG.',
'CIPROFLOXACINO_.MG.',
'CLARITROMICINA_.MG.',
'CLINDAMICINA_.MG.',
'CLOXACILINA_.MG.',
'COTRIMOXAZOL_FORTE_.SULFAMETOXAZOL_.UND.',
'COTRIMOXAZOL.SULFAMETOXAZOL_.MG.',
'COTRIMOXAZOL.SULFAMETOXAZOL_.UND.',
'DAPTOMICINA_.MG.',
'DORIPENEM_.MG.',
'DOXICICLINA_.MG.',
'ERITROMICINA_.MG.',
'ERTAPENEM_.MG.',
'FOSFOMICINA_.MG.',
'GENTAMICINA_.MG.',
'IMIPENEM_.MG.',
'LEVOFLOXACINO_.MG.',
'LINEZOLID_.MG.',
'MEROPENEM_.MG.',
'METRONIDAZOL_.MG.',
'METRONIDAZOL_COMP_.MG.',
'MOXIFLOXACINO_.MG.',
'NORFLOXACINO_.MG.',
'PIPERACILINA_.MG.',
'RIFABUTINA_.MG.',
'RIFAMPICINA_.MG.',
'SULFADIAZINA_.MG.',
'TEICOPLANINA_.MG.',
'TIGECICLINA_.MG.',
'TOBRAMICINA_.MG.',
'TOBRAMICINA_NEB_.MG.',
'VANCOMICINA_.MG.']

# Load data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_challenge.csv')

# ------------------------------------------------
# Preprocessing
# ------------------------------------------------

# General preprocessing
df_train_cln = preprocessing(df_train)
df_test_cln = preprocessing(df_test)

# PCA
resulting_features_names = ['PC1_DIAGNOSTIC', 'PC2_DIAGNOSTIC']
pc_diagnosis = PCA_r(df_train_cln, features_diagnostic, 2, resulting_features_names)

resulting_features_names = ['PC1_ANTIBIOTIC', 'PC2_ANTIBIOTIC']
pc_antibiotics = PCA_r(df_train_cln, features_antibiotic, 2, resulting_features_names)

# Adding PCA columns to original dataset
df_train_cln = pd.concat([df_train_cln, pc_diagnosis, pc_antibiotics], axis=1)

# ------------------------------------------------
# Modelling
# ------------------------------------------------
FLAG = 'MDR'

# extract features
features = extract_features(df_train_cln, dummies_include=True)

# Model 1 - ROOM LIST MODEL
# ------------------------------------------------
df_rooms = df_train_cln[['ID', FLAG, 'room_list']]

for idx, row in df_rooms.iterrows():
    try:
        for room in row['room_list'].split(","):
            room_col = room.strip()
            df_rooms.loc[idx, room_col] = 1
    except:
        if np.isnan(row['room_list']):
            df_rooms.loc[idx, 'UNK'] = 1
        else:
            room_col = row['room_list'].strip()
            df_rooms.loc[idx, room_col] = 1

df_rooms = df_rooms.fillna(0)

list_rooms = list(df_rooms.columns.values[3:])


df_test_rooms = generate_rooms(df_test_cln, list_rooms)

df_rooms.drop('room_list', axis=1, inplace=True)
df_rooms.drop('ID', axis=1, inplace=True)

features = list(df_rooms.columns.values)
predictors = features
predictors.pop(0) # pop flag

# train / test split
X = df_rooms[features].values
y = df_rooms[FLAG].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

train = pd.DataFrame(X_train, columns=[features])
train[FLAG] = list(y_train)

test = pd.DataFrame(X_test, columns=[features])
test[FLAG] = y_test


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=1,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 reg_alpha=0,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb1, df_features = xgboost_fit(xgb1, train, test, y_train, y_test, predictors, target=FLAG)

y_room = xgb1.predict_proba(df_rooms[predictors])[:,1]

df_train_cln['y_room'] = y_room
