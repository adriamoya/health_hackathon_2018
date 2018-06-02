import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def preprocessing(df):
    """ Preprocessing """

    # Days between
    df['d_days_between'] = df['days_between'].apply(lambda x: 0 if np.isnan(x) else 1)
    df['d_days_after_anti'] = df['days_after_anti'].apply(lambda x: 0 if np.isnan(x) else 1)
    df.fillna({'days_between':0,'days_after_anti':0 },inplace=True)

    # List of rooms
    for idx, row in df.iterrows():
        try:
            num_rooms = len(row['room_list'].split(","))
        except:
            if np.isnan(row['room_list']):
                num_rooms = 0
            else:
                num_rooms = 1
        df.loc[idx, 'num_rooms'] = num_rooms

    df.drop('room_list', axis=1, inplace=True)
    df['num_rooms_b'] = df['num_rooms'].apply(lambda x: x if x <= 2 else 3)
    df.drop('num_rooms', axis=1, inplace=True)

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

    return df


def extract_features(df):
    """ Extract features """

    features = df.columns[(~df.columns.str.contains('_.MG'))
    & (~df.columns.str.contains('_.UND'))
    & (~df.columns.str.contains('ID'))
    & (~df.columns.str.contains('NHC'))
    & (~df.columns.str.contains('start_'))
    & (~df.columns.str.contains('Gender'))
    & (~df.columns.str.contains('birth_year'))
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
