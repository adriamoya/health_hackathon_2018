
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV #Perforing grid search
from sklearn.model_selection import KFold, StratifiedKFold

from preprocessing.preprocessing import preprocessing, extract_features, PCA_r, generate_rooms

import xgboost as xgb
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

#df_train_cln.drop(df_train_cln[df_train_cln['ID']=='1885-1'].index, inplace=True)
df_train_cln[df_train_cln['ID']=='1885-1'].index
df_train_cln.loc[1439, 'days_between'] = int(400)

df_train_cln[df_train_cln['ID']=='1885-1']

# PCA
resulting_features_names = ['PC1_DIAGNOSTIC', 'PC2_DIAGNOSTIC']
pca, min_max_scaler,pc_diagnosis  = PCA_r(df_train_cln, features_diagnostic, 2, resulting_features_names)
X_test = df_test_cln.loc[:, features_diagnostic].values
X_test = min_max_scaler.transform(X_test)
principalComponents = pca.transform(X_test)
principalDf = pd.DataFrame(data = principalComponents
              , columns = [resulting_features_names[0], resulting_features_names[1]])
df_test_cln = pd.concat([df_test_cln, principalDf], axis=1)


resulting_features_names = ['PC1_ANTIBIOTIC', 'PC2_ANTIBIOTIC']
pca, min_max_scaler, pc_antibiotics = PCA_r(df_train_cln, features_antibiotic, 2, resulting_features_names)
X_test = df_test_cln.loc[:, features_antibiotic].values
X_test = min_max_scaler.transform(X_test)
principalComponents = pca.transform(X_test)
principalDf = pd.DataFrame(data = principalComponents
              , columns = [resulting_features_names[0], resulting_features_names[1]])
df_test_cln = pd.concat([df_test_cln, principalDf], axis=1)

# Adding PCA columns to original dataset
df_train_cln = pd.concat([df_train_cln, pc_diagnosis, pc_antibiotics], axis=1)



# ------------------------------------------------
# Modelling
# ------------------------------------------------
FLAG = 'MDR'

# extract features
features = extract_features(df_train_cln, dummies_include=False)

predictors = features
predictors.pop(0) # pop flag

# train / test split
X = df_train_cln[features].values
y = df_train_cln[FLAG].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train = pd.DataFrame(X_train, columns=[features])
train[FLAG] = list(y_train)

test = pd.DataFrame(X_test, columns=[features])
test[FLAG] = y_test

# standardization
scaler = StandardScaler()
dtrain = pd.DataFrame(scaler.fit_transform(train[predictors]), columns=features)
dtest = pd.DataFrame(scaler.transform(test[predictors]), columns=features)
 #
 # learning_rate =0.1,
 # n_estimators=1000,
 # max_depth=5,
 # min_child_weight=1,
 # gamma=0,
 # subsample=0.8,
 # colsample_bytree=0.8,
 # objective= 'binary:logistic',
 # reg_alpha=0,
 # nthread=4,
 # scale_pos_weight=1,
 # seed=29)

# xgboost model
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
 scale_pos_weight=1, # 1
 seed=26)

xgb1, df_features = xgboost_fit(xgb1, dtrain, dtest, y_train, y_test, predictors, target=FLAG)

predictors = df_features[:24]['feature'].values


# train / test split
X = df_train_cln[predictors].values
y = df_train_cln[FLAG].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train = pd.DataFrame(X_train, columns=[predictors])
train[FLAG] = list(y_train)

test = pd.DataFrame(X_test, columns=[predictors])
test[FLAG] = y_test

# standardization
scaler = StandardScaler()
dtrain = pd.DataFrame(scaler.fit_transform(train[predictors]), columns=predictors)
dtest = pd.DataFrame(scaler.transform(test[predictors]), columns=predictors)
 #
 # learning_rate =0.1,
 # n_estimators=1000,
 # max_depth=5,
 # min_child_weight=1,
 # gamma=0,
 # subsample=0.8,
 # colsample_bytree=0.8,
 # objective= 'binary:logistic',
 # reg_alpha=0,
 # nthread=4,
 # scale_pos_weight=1,
 # seed=29)

# xgboost model
xgb1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=2,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 reg_alpha=1,
 nthread=4,
 scale_pos_weight=1, # 1
 seed=26)

xgb1, df_features = xgboost_fit(xgb1, dtrain, dtest, y_train, y_test, predictors, target=FLAG)

# ------------------------------------------------
# Submission
# ------------------------------------------------

# Predict using xgb1
df_subm = pd.DataFrame(scaler.transform(df_test_cln[predictors]), columns=predictors)
y_subm = xgb1.predict_proba(df_subm[predictors])[:,1]
df_subm['pred'] = y_subm

submission = df_test_cln[['ID']]
submission['MDR'] = y_subm

# Write the final CSV file.
submission.to_csv("./submissions/xgboost_pca.csv", encoding='utf-8', index=False)
