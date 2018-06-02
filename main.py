
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import preprocessing, extract_features

from xgboost import XGBClassifier
from models.xgboost_model import xgboost_fit

from matplotlib import pyplot
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

# Load data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_challenge.csv')

# Preprocessing
# ------------------------------------------------
df_train_cln = preprocessing(df_train)
df_test_cln = preprocessing(df_test)

# Modelling
# ------------------------------------------------

FLAG = 'MDR'

# extract features
features = extract_features(df_train_cln)

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
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgboost_fit(xgb1, train, test, predictors, target=FLAG)
