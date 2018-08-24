import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    df = pd.read_csv('datasets/studentscores.csv')
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

    lr = LinearRegression()
    lr = lr.fit(X_train, Y_train)

    plt.subplot(2, 1, 1)
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, lr.predict(X_train), color='blue')

    plt.subplot(2, 1, 2)
    plt.scatter(X_test , Y_test, color='red')
    plt.plot(X_test , lr.predict(X_test), color='blue')

    plt.show()
