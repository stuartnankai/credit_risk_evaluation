import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


def outlier_processing(df, col):
    s = df[col]
    oneQuoter = s.quantile(0.25)
    threeQuote = s.quantile(0.75)
    irq = threeQuote - oneQuoter
    min = oneQuoter - 1.5 * irq
    max = threeQuote + 1.5 * irq
    df = df[df[col] <= max]
    df = df[df[col] >= min]
    return df


if __name__ == '__main__':
    df = pd.read_csv('MissingData.csv')
    # check the age
    df_new = df[df['age'] > 0]
    df_new = df_new[df_new['NumberOfTime30-59DaysPastDueNotWorse'] < 90]  # outlier
    df_new['SeriousDlqin2yrs'] = 1 - df_new['SeriousDlqin2yrs']
    Y = df_new['SeriousDlqin2yrs']
    X = df_new.ix[:, 1:]
    # CV part
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    # Build the training and testing data
    train.to_csv('TrainData.csv', index=False)
    test.to_csv('TestData.csv', index=False)
    # print(train.shape)
    # print(test.shape)
