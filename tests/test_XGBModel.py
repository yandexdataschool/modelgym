import pytest
import numpy as np
import xgboost
import pandas as pd
import modelgym
import pickle
from sklearn.model_selection import train_test_split
import sklearn.datasets.data

TEST_SIZE = 0.5
N_CV_SPLITS = 2
N_ROWS = 1000
def_params = {}
TEST_PARAMS = ["classification", "range", "regression"]


def test_preprocess_params():
    global def_params
    test_params = ["classification", "regression"]
    for par1 in test_params:
        model = modelgym.XGBModel(learning_task=par1)
        if def_params == {}:
            def_params = model.default_params
        k = len(model.default_params)
        if model.learning_task == "classification":
            model.default_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif model.learning_task == "regression":
            model.default_params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        model.default_params['max_depth'] = int(model.default_params['max_depth'])
        assert test_params.__contains__(model.learning_task)
        assert len(model.default_params) != k + 3
    return 0


def test_convert_to_dataset(preprocess):
    X_train, X_test, y_train, y_test = preprocess
    # iris_data=preprocess
    global TEST_PARAMS
    dtrain = xgboost.DMatrix(X_train, y_train)
    dtest = xgboost.DMatrix(X_test, y_test)
    model = modelgym.XGBModel(learning_task=TEST_PARAMS[0])
    dexample = model.convert_to_dataset(data=X_train, label=y_train)
    # compare dtrain and dexample True
    # compare dtest and dexample False
    pass


def test_fit(preprocess):
    X_train, X_test, y_train, y_test = preprocess
    dtrain = xgboost.DMatrix(X_train, y_train)
    dtest = xgboost.DMatrix(X_test, y_test)
    evals_result = {}
    params = def_params
    result = xgboost.train(dtrain=dtrain, params=params, evals=[(dtest, 'test')], evals_result=evals_result,
                           num_boost_round=params['n_estimators'], verbose_eval=False)
    assert result.score.call_count == 1


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


def test_XGBModel():
    global TEST_PARAMS
    for par1 in TEST_PARAMS:
        try:
            model = modelgym.XGBModel(learning_task=par1)
        except ValueError:
            print(par1, "not expected")
        param = def_params
        if model.learning_task == "classification":
            param.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif model.learning_task == "regression":
            param.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        param['max_depth'] = int(param['max_depth'])
        assert model.default_params == param
