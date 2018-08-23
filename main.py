import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    # data load
    df = pd.read_csv('datasets/Data.csv')
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    # handle missing data
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

    # encode data
    label_encoder_X = LabelEncoder()
    X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    onehotencoder = OneHotEncoder(categorical_features=[0])
    X = onehotencoder.fit_transform(X).toarray()

    # data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # data scaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    print (X_train, X_test)
