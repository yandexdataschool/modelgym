import pytest
import numpy as np
import xgboost
import pandas as pd
import modelgym
import pickle
from sklearn.model_selection import train_test_split
import sklearn.datasets.data
from modelgym.util import split_and_preprocess

TEST_SIZE = 0.5
N_CV_SPLITS = 2
N_ROWS = 1000
TEST_PARAMS = ["classification", "range", "regression"]


def test_preprocess_params():
    for par1 in TEST_PARAMS:
        try:
            model = modelgym.XGBModel(learning_task=par1)
        except ValueError:
            print(par1, "not expected")
        k = len(model.default_params)
        model.preprocess_params(model.default_params)
        assert TEST_PARAMS.__contains__(model.learning_task)
        assert len(model.default_params) != k + 3
    return 0


def test_convert_to_dataset(preprocess):
    X_train, X_test, y_train, y_test = preprocess
    # print ('xtrain',X_train)
    # print('xtest',X_test)
    # print('ytrain',y_train)

    dtrain = xgboost.DMatrix(X_train, y_train)
    dtest = xgboost.DMatrix(X_test, y_test)
    model = modelgym.XGBModel(learning_task=TEST_PARAMS[0])
    dexample = model.convert_to_dataset(data=X_train, label=y_train)
    # compare dtrain and dexample True
    # compare dtest and dexample False
    pass


def test_fit(preprocess):
    X_train, X_test, y_train, y_test = preprocess
    evals_result = []
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = modelgym.XGBModel(learning_task="classification")
    params = model.preprocess_params(model.default_params)
    for dtrain, dtest in cv_pairs:
        _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
        _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
        _, evals_result = model.fit(params, _dtrain, _dtest, params['n_estimators'])
        evals_results.append(evals_result)
    assert len(evals_result) != 0


def test_predict():
    assert 0 == 0


@pytest.fixture
def read_data():
    iris = sklearn.datasets.load_iris()
    return iris  # data and target


@pytest.fixture
def preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test
