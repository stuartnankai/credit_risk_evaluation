import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


# for the missing values, fill them in by using RF.
def replace_missing(df):
    # get the numerical features
    num_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]

    # print("This is num_df: ", num_df)
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
df_new = df_new[df_new['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
# for new SeriousDlqin2yrs, 1: good, 0: bad
df_new['SeriousDlqin2yrs']=1-df_new['SeriousDlqin2yrs']


#Exploratory Data Analysis (EDA)

x = df_new['age'].value_counts().sort_index().plot.bar()
plt.show()

# print("This is x: ", x)

# # CV
# Y = df_new['SeriousDlqin2yrs']  # As label
# X = df_new.ix[:, 1:] # training data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# # print("This is X_test: ", X_test )
# train = pd.concat([Y_train, X_train], axis=1) # combine data
# test = pd.concat([Y_test, X_test], axis=1)
# clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count() #
# train.to_csv('TrainData.csv',index=False)
# test.to_csv('TestData.csv',index=False)

