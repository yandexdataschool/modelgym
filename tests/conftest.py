import pandas as pd
import pytest
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.5


@pytest.fixture(scope="session")
def read_titanic():
    print("\nReading data from CSV...")
    data_train = pd.read_csv('data/train.csv')
    # data_test = pd.read_csv('data/test.csv')
    return data_train


@pytest.fixture(scope="session")
def preprocess_data(read_titanic):
    def simplify_fares(df):
        # Create an imputer object
        fare_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        # Fit the imputer object on the training data
        fare_imputer.fit(df['Fare'].reshape(-1, 1))
        # Apply the imputer object to the training and test data
        df['Fare'] = fare_imputer.transform(df['Fare'].reshape(-1, 1))
        return df

    def simplify_sex(df):
        # Create an encoder
        sex_encoder = preprocessing.LabelEncoder()
        # Fit the encoder to the train data so it knows that male = 1
        sex_encoder.fit(df['Sex'])
        # Apply the encoder to the training data
        df['Sex'] = sex_encoder.transform(df['Sex'])
        return df

    def simplify_ages(df):
        # Create an imputer object
        age_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        # Fit the imputer object on the training data
        age_imputer.fit(df['Age'].reshape(-1, 1))
        # Apply the imputer object to the training and test data
        df['Age'] = age_imputer.transform(df['Age'].reshape(-1, 1))
        return df

    print("Processing data from CSV...")
    data_train = read_titanic
    data_train = data_train[['Survived', 'Age', 'SibSp', 'Fare', 'Sex']]
    data_train = simplify_fares(data_train)
    data_train = simplify_sex(data_train)
    data_train.Age = simplify_ages(data_train)
    # data_test = data_test[['PassengerId', 'Age']]
    X_all = data_train.drop(['Survived'], axis=1)
    y_all = data_train['Survived']
    X_all = X_all.values
    y_all = y_all.values

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def read_data():
    iris = load_iris()
    return iris  # data and target


@pytest.fixture()
def generate_trials():
    import math
    from hyperopt import fmin, tpe, hp, Trials
    trials = Trials()
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
    yield trials
