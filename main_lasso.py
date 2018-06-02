
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV #Perforing grid search
from sklearn import linear_model

from preprocessing.preprocessing import preprocessing, extract_features, PCA_r

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
features = extract_features(df_train_cln, dummies_include=False)

predictors = features
predictors.pop(0) # pop flag

features = [
FLAG,
'age',
'prev_hospital_stay',
'days_between',
'days_after_anti',
'days_in_hospital',
'num_consult',
'share_room_MDR',
'hospital_stay_w_FN',
'days_neutropenic_wo_fn',
'num_rooms_b',
'num_movements',
'gender__female',
'emergency',
'antibiotic_count',
'dummy_Cancer.linfoproliferativo',
'cito_group_2',
'cito_group_3',
'mucositis',
'dummy_Mieloma.like',
'Alo_TP']
predictors = features
predictors.pop(0) # pop flag


# train / test split
X = df_train_cln[predictors].values
y = df_train_cln[FLAG].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

train = pd.DataFrame(X_train, columns=[predictors])
train[FLAG] = list(y_train)

test = pd.DataFrame(X_test, columns=[predictors])
test[FLAG] = y_test

# standardization
scaler = StandardScaler()
dtrain = pd.DataFrame(scaler.fit_transform(train[predictors]), columns=predictors)
dtest = pd.DataFrame(scaler.transform(test[predictors]), columns=predictors)

# LassoCV

alphas = [0.0000001, 0.00001, 0.0001, 0.01, 0.1, 1, 10]

for alpha in alphas:

    clf = linear_model.LassoCV(alphas=[alpha])

    clf.fit(dtrain[predictors].values, y_train)
    pred_train = clf.predict(dtrain[predictors].values)
    print( "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, pred_train))

    pred_test = clf.predict(dtest[predictors].values)
    print( "AUC Score (Test): %f" % metrics.roc_auc_score(y_test, pred_test))


clf = linear_model.LassoCV(alphas=[0.0001])

clf.fit(dtrain[predictors].values, y_train)
pred_train = clf.predict(dtrain[predictors].values)
print( "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, pred_train))

pred_test = clf.predict(dtest[predictors].values)
print( "AUC Score (Test): %f" % metrics.roc_auc_score(y_test, pred_test))


# ------------------------------------------------
# Submission
# ------------------------------------------------
df_subm = pd.DataFrame(scaler.transform(df_test_cln[predictors]), columns=predictors)

y_subm = clf.predict(df_subm[predictors])

df_subm['pred'] = y_subm

submission = df_test_cln[['ID']]
submission['MDR'] = y_subm

# Write the final CSV file.
submission.to_csv("./submissions/lasso_0000001_20.csv", encoding='utf-8', index=False)
