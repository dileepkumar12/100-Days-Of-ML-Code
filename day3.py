import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    df = pd.read_csv('datasets/50_Startups.csv')
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

    lr = LinearRegression()
    lr = lr.fit(X_train, Y_train)

    print(lr.predict(X_test))
    print(Y_test)
