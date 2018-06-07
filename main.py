
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

# PCA
resulting_features_names = ['PC1_DIAGNOSTIC', 'PC2_DIAGNOSTIC']
pc_diagnosis = PCA_r(df_train_cln, features_diagnostic, 2, resulting_features_names)

resulting_features_names = ['PC1_ANTIBIOTIC', 'PC2_ANTIBIOTIC']
pc_antibiotics = PCA_r(df_train_cln, features_antibiotic, 2, resulting_features_names)

# Adding PCA columns to original dataset
df_train_cln = pd.concat([df_train_cln, pc_diagnosis, pc_antibiotics], axis=1)

# Generate rooms
df_train_rooms = df_train_cln[['ID', 'room_list']]

for idx, row in df_train_rooms.iterrows():
    df_train_rooms.loc[idx, 'UNK'] = 0
    try:
        for room in row['room_list'].split(","):
            room_col = room.strip()
            df_train_rooms.loc[idx, room_col] = 1
    except:
        if np.isnan(row['room_list']):
            df_train_rooms.loc[idx, 'UNK'] += 1
        else:
            room_col = row['room_list'].strip()
            df_train_rooms.loc[idx, room_col] = 1

df_train_rooms = df_train_rooms.fillna(0)

list_rooms = list(df_train_rooms.columns.values[2:])

df_test_rooms = generate_rooms(df_test_cln, list_rooms)

df_train_rooms.drop('ID', axis=1, inplace=True)
df_train_rooms.drop('room_list', axis=1, inplace=True)
df_test_rooms.drop('ID', axis=1, inplace=True)
df_test_rooms.drop('room_list', axis=1, inplace=True)

df_train_cln = pd.concat([df_train_cln, df_train_rooms], axis=1)
df_test_cln = pd.concat([df_test_cln, df_test_rooms], axis=1)

df_train_cln.drop('room_list', axis=1, inplace=True)
df_test_cln.drop('room_list', axis=1, inplace=True)

df_train_cln.shape
df_test_cln.shape

# ------------------------------------------------
# Modelling
# ------------------------------------------------
FLAG = 'MDR'

# extract features
features = extract_features(df_train_cln, dummies_include=True)

predictors = features
predictors.pop(0) # pop flag

# train / test split
X = df_train_cln[features].values
y = df_train_cln[FLAG].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

train = pd.DataFrame(X_train, columns=[features])
train[FLAG] = list(y_train)

test = pd.DataFrame(X_test, columns=[features])
test[FLAG] = y_test

# standardization
scaler = StandardScaler()
dtrain = pd.DataFrame(scaler.fit_transform(train[predictors]), columns=features)
dtest = pd.DataFrame(scaler.transform(test[predictors]), columns=features)

# weight
num_flags = sum(y_train)

w = []
n1 = sum(y_train)
n = len(y_train)
for y_item in y_train:
    if y_item == 1:
        w.append(1)
    elif y_item == 0:
        w.append(n1/(n-n1))

# xgboost model
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 reg_alpha=0,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

dtrain.describe()

xgb1, df_features = xgboost_fit(xgb1, dtrain, dtest, y_train, y_test, predictors, target=FLAG)


# Grid search
#
# params = {
#         'min_child_weight': [1, 3, 5],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'learning_rate': [0.001, 0.01, 0.1]
#         }
#
# gsearch1 = GridSearchCV(
#     estimator = XGBClassifier(
#         learning_rate =0.1,
#         n_estimators=1000,
#         max_depth=4,
#         min_child_weight=1,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective= 'binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27,
#         verbose_eval=0),
#     param_grid = params, scoring='roc_auc', n_jobs=4, iid=False, cv=5, return_train_score=False)
#
# gsearch1.fit(dtrain[predictors].values, y_train)
# xgb2, df_features = xgboost_fit(gsearch1.best_estimator_, dtrain, dtest, y_train, y_test, predictors, target=FLAG)
# gsearch1.best_params_, gsearch1.best_score_
#
# xgb2.predict_proba()


# ------------------------------------------------
# Folds (stacking 4 xgboosts)
# ------------------------------------------------
#
# _X = pd.DataFrame(scaler.fit_transform(df_train_cln[features].values), columns=features).values
# _X_test = pd.DataFrame(scaler.transform(df_test_cln[features].values), columns=features).values
#
# n_folds = 4
# kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=0)
#
# S_test_temp = np.zeros((_X_test.shape[0], n_folds))
#
# # Loop del stacking
# for fold_counter, (tr_index, te_index) in enumerate(kf.split(_X, y)):
#
#     # Loop across folds
#     print('\nFold %s' % fold_counter)
#     print('-'*80)
#
#     # Split data and target
#     X_tr = _X[tr_index]
#     y_tr = y[tr_index]
#     X_te = _X[te_index]
#     y_te = y[te_index]
#
#     xgb1 = XGBClassifier(
#      learning_rate =0.1,
#      n_estimators=1000,
#      max_depth=4,
#      min_child_weight=1,
#      gamma=1,
#      subsample=0.8,
#      colsample_bytree=0.8,
#      objective= 'binary:logistic',
#      reg_alpha=0,
#      nthread=4,
#      scale_pos_weight=1,
#      seed=27)
#
#     xgb_param = xgb1.get_xgb_params()
#     xgtrain = xgb.DMatrix(X_tr, label=y_tr)
#     cvresult = xgb.cv(
#         xgb_param,
#         xgtrain,
#         num_boost_round=xgb1.get_params()['n_estimators'],
#         nfold=4,
#         metrics='auc',
#         early_stopping_rounds=50,
#         verbose_eval=0)
#     xgb1.set_params(n_estimators=cvresult.shape[0])
#
#     xgb1.fit(X_tr, y_tr,eval_metric='auc')
#
#     # Predict training set:
#     dtrain_predictions = xgb1.predict(X_tr)
#     dtrain_predprob = xgb1.predict_proba(X_tr)[:,1]
#
#     # Print model report:
#     print( "\nModel Report (Train)")
#     print( "Accuracy : %.4g" % metrics.accuracy_score(y_tr, dtrain_predictions))
#     print( "AUC Score: %f" % metrics.roc_auc_score(y_tr, dtrain_predprob))
#
#     # Predict validation set:
#     dtest_predprob = xgb1.predict_proba(X_te)[:,1]
#
#     # Print model report:
#     print( "\nModel Report (Test)")
#     print( "AUC Score: %f" % metrics.roc_auc_score(y_te, dtest_predprob))
#
#     S_test_temp[:, fold_counter] = xgb1.predict_proba(_X_test)[:,1]
#
# y_subm = S_test_temp[:, -1]
# # average folds scores
# S_test_temp[:, 1:].mean(axis=1)
# y_subm = S_test_temp.mean(axis=1)
#


# xgboost with top features (deprecated)
# ------------------------------------------------
#
predictors = list(df_features[:40].feature.values)

# standardization
scaler = StandardScaler()
dtrain = pd.DataFrame(scaler.fit_transform(train[predictors]), columns=predictors)
dtest = pd.DataFrame(scaler.transform(test[predictors]), columns=predictors)


xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=5,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 reg_alpha=0,
 nthread=4,
 #scale_pos_weight=1,
 seed=27)

xgb2, df_features = xgboost_fit(xgb2, dtrain, dtest, y_train, y_test, predictors, target=FLAG)


# ------------------------------------------------
# Submission
# ------------------------------------------------

# Predict using xgb1
df_subm = pd.DataFrame(scaler.transform(df_test_cln[predictors]), columns=predictors)
y_subm = xgb2.predict_proba(df_subm[predictors])[:,1]
df_subm['pred'] = y_subm

submission = df_test_cln[['ID']]
submission['MDR'] = y_subm

# Write the final CSV file.
submission.to_csv("./submissions/xgboost_rooms.csv", encoding='utf-8', index=False)
