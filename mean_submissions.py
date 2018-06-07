
import pandas as pd


df1 = pd.read_csv('./submissions/first-submission.csv')
df2 = pd.read_csv('./submissions/fold_mean_xgboost.csv')
df3 = pd.read_csv('./submissions/mean_submissions_nolasso.csv')
df4 = pd.read_csv('./submissions/xgboost_scale_pos_weight1.csv')
df5 = pd.read_csv('./submissions/xgboost_scale_pos_weight1_incl.csv')
df6 = pd.read_csv('./submissions/xgboost_pca.csv')
#df3 = pd.read_csv('./submissions/lasso_0000001_20.csv')

df2.drop('ID', axis=1, inplace=True)
df3.drop('ID', axis=1, inplace=True)
df4.drop('ID', axis=1, inplace=True)
df5.drop('ID', axis=1, inplace=True)
df6.drop('ID', axis=1, inplace=True)

df_total = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)

df_total.drop('ID', axis=1, inplace=True)

df_subm = pd.concat([df1['ID'], df_total.mean(axis=1)], axis=1).rename(columns={0:"MDR"})
df_subm.shape

df_subm.to_csv("./submissions/total_combination.csv", encoding='utf-8', index=False)
