import pandas as pd
import numpy as np
from pandas import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


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
    # replace the missing values
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


def discretization(label, input, n):
    r = 0
    positive = label.sum()  # good clients
    negative = label.count() - positive  # bad clients
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": input, "Y": label, "Bucket": pd.qcut(input, n)})
        print("*" * 100)
        print("This is d1: ", d1)
        d2 = d1.groupby("Bucket", as_index=True)
        r, p = stats.spearmanr(d2.mean().input, d2.mean().label)
        n = n - 1
    d3 = pd.DataFrame(d2.input.min(), columns=['min'])
    d3['min'] = d2.min().input
    d3['max'] = d2.max().input
    d3['sum'] = d2.sum().label
    d3['total'] = d2.count().label
    d3['rate'] = d2.mean().label
    d3['woe'] = np.log(d3['rate']/(1-d3['rate']))/(positive/negative)
    d4 = (d3.sort_index(by='min')).reset_index(drop=True)
    print("*"*100)
    print("This is d4: ", d4)
    return d4

if __name__ == '__main__':


    # loading data
    df = pd.read_csv('cs-training.csv')
    # print("This is df: ", df.head())
    del df['Id']
    df.describe().to_csv('Data_description.csv')
    # Solve the missing value issue
    df_new = replace_missing(df)
    # Drop the missing values
    df_new = df_new.dropna()
    # Drop the duplicates
    df_new = df_new.drop_duplicates()
    df_new.to_csv('MissingData.csv', index=False)

    # Exploratory Data Analysis (EDA)
    # age_plot = df_new['age'].value_counts().sort_index().plot.bar()
    # plt.show()
