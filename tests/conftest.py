import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import math

TEST_SIZE = 0.5


@pytest.fixture(scope="session")
def read_cancer():
    print("\nReading data from dataset Breast Cancer...")
    return load_breast_cancer()


@pytest.fixture(scope="session")
def preprocess_data(read_cancer):
    print("Processing data...")
    data = read_cancer
    X_all, y_all = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def read_data():
    print("\nReading data from dataset Iris...")
    return load_iris()  # data and target


@pytest.fixture()
def generate_trials():
    from hyperopt import fmin, tpe, hp, Trials
    trials = Trials()
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
    yield trials
