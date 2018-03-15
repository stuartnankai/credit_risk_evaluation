import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# for the missing values, fill them in by using RF.
def replace_missing(df):
    # get the numerical features
    num_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]

    print("This is num_df: ", num_df)
    # split the income feature,
    known = num_df[num_df.MonthlyIncome.notnull()].as_matrix()
    unknown = num_df[num_df.MonthlyIncome.isnull()].as_matrix()
    # training data
    X = known[:, 1:]
    # labels
    y = known[:, 0]
    # fit the model
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # using model to predict
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # replace the missing values
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


# loading data
df = pd.read_csv('cs-training.csv')
# print("This is df: ", df.head())
del df['Id']
# df.describe().to_csv('Data_description.csv')
# Solve the missing value issue
df_new = replace_missing(df)
# Drop the missing values
df_new = df_new.dropna()
# Drop the duplicates
df_new = df_new.drop_duplicates()
# df_new.to_csv('MissingData.csv', index=False)

# Check the outliers
df_new = df_new[df_new['age'] > 0]



