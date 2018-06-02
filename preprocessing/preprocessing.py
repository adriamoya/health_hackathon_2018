import numpy as np
import pandas as pd
import datetime

from sklearn.decomposition import PCA
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

DANGER_ROOMS = [
'I0570',
'E01503',
'E02401',
'G06514',
'G06506',
'G06513',
'G02409',
'G06510']

def woe_fit(df, col, FLAG):
    """ Fit woes """
    col = 'Past_positive__Both'
    n11 = df[df[col]==1][FLAG].sum()
    n1 = df[df[col]==1][FLAG].count()
    woe_1 = np.log((n1-n11)/n11)

    n01 = df[df[col]==0][FLAG].sum()
    n0 = df[df[col]==0][FLAG].count()
    woe_0 = np.log((n0-n01)/n01)

    df[col+"_w"] = df[col].apply(lambda x: woe_1 if x > 0 else woe_0 )

    return df

def preprocessing(df):
    """ Preprocessing """

    # Days between
    df['d_days_between'] = df['days_between'].apply(lambda x: 0 if np.isnan(x) else 1)
    df['d_days_after_anti'] = df['days_after_anti'].apply(lambda x: 0 if np.isnan(x) else 1)
    df.fillna({'days_between':0,'days_after_anti':0 },inplace=True)

    # List of rooms
    for idx, row in df.iterrows():
        df.loc[idx, 'danger_room'] = 0
        try:
            rooms = row['room_list'].split(",")
            for room in rooms:
                if room in DANGER_ROOMS:
                    df.loc[idx, 'danger_room'] = 1
            num_rooms = len(rooms)
        except:
            if np.isnan(row['room_list']):
                num_rooms = 0
            else:
                num_rooms = 1
                if row['room_list'].strip() in DANGER_ROOMS:
                    df.loc[idx, 'danger_room'] = 1

        df.loc[idx, 'num_rooms'] = num_rooms

    # df.drop('room_list', axis=1, inplace=True)
    df['num_rooms_b'] = df['num_rooms'].apply(lambda x: x if x <= 2 else 3)
    df.drop('num_rooms', axis=1, inplace=True)


    # Times
    df['num_veces_enfermo'] = df['ID'].map(lambda x: str(x)[-1])

    # Gender dummies
    df_gender_dum = pd.get_dummies(df.Gender , prefix='gender_')
    df = pd.concat([df, df_gender_dum], axis=1)

    # Past positive results dummies
    df_Past_positive_dum = pd.get_dummies(df.Past_positive_result_from , prefix='Past_positive_')
    df = pd.concat([df, df_Past_positive_dum], axis=1)

    def dates(df,days_neutropenic_wo_fn,dummie_days_neutropenic_wo_fn, age):
        """ Create dates """

        dates_columns = ["start_neutropenico", "start_FN", "birth_year"]
        for c in dates_columns:
            if c != "birth_year":
                df[c] = pd.to_datetime(df[c],format='%Y-%m-%d')
            else:
                df[c] = pd.to_datetime(df[c],format="%Y")

        df[days_neutropenic_wo_fn] = df["start_FN"] - df["start_neutropenico"]
        df[age] = df["start_FN"].dt.year - df["birth_year"].dt.year

        #conver to integer
        df[days_neutropenic_wo_fn] = df[days_neutropenic_wo_fn].dt.days

        #dummie if patient got FN after few days of having neutropenic status
        df[dummie_days_neutropenic_wo_fn] = np.where(df[days_neutropenic_wo_fn]>0, 1, 0)

        return df

    dates(df,'days_neutropenic_wo_fn','dummie_days_neutropenic_wo_fn', 'age')

    # Month
    df['month'] = df['start_neutropenico'].apply(lambda x: x.month)
    
    return df


def extract_features(df, dummies_include=False):
    """ Extract features """

    if dummies_include:
        features = df.columns[(~df.columns.str.contains('_.MG'))
        & (~df.columns.str.contains('_.UND'))
        & (~df.columns.str.contains('ID'))
        & (~df.columns.str.contains('NHC'))
        & (~df.columns.str.contains('start_'))
        & (~df.columns.str.contains('Gender'))
        & (~df.columns.str.contains('birth_year'))
        & (~df.columns.str.contains('PC1'))
        & (~df.columns.str.contains('PC2'))
        & (~df.columns.str.contains('Past_positive_result_from'))
        ].values # if index are necessary, remove .values

    else:
        features = df.columns[(~df.columns.str.contains('_.MG'))
        & (~df.columns.str.contains('_.UND'))
        & (~df.columns.str.contains('ID'))
        & (~df.columns.str.contains('NHC'))
        & (~df.columns.str.contains('start_'))
        & (~df.columns.str.contains('Gender'))
        & (~df.columns.str.contains('birth_year'))
        & (~df.columns.str.contains('dummy_'))
        & (~df.columns.str.contains('Past_positive_result_from'))
        ].values # if index are necessary, remove .values

    return list(features)


def PCA_r(df, features, num_comp, resulting_features_names):
    """ PCA """
    # Separating out the features
    X = df.loc[:, features].values
    X = min_max_scaler.fit_transform(X)
    # Separating out the target
    y = df[['MDR']]

    pca = PCA(n_components=num_comp)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = [resulting_features_names[0], resulting_features_names[1]])

    return principalDf
