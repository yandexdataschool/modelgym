import pytest
import numpy as np
import xgboost
import pandas as pd
import modelgym
import pickle
from sklearn.model_selection import train_test_split
from modelgym.util import split_and_preprocess
from unittest.mock import Mock

TEST_SIZE = 0.5
N_CV_SPLITS = 2
def_params = {'base_score': 0.5,
              'colsample_bylevel': 1,
              'colsample_bytree': 1,
              'gamma': 0,
              'learning_rate': 0.1,
              'max_delta_step': 0,
              'max_depth': 3,
              'min_child_weight': 1,
              'missing': None,
              'n_estimators': 100,
              'nthread': -1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'scale_pos_weight': 1,
              'seed': 0,
              'subsample': 1}


def test_preprocess_params():
    test_params = ["classification", "regression"]
    for par1 in test_params:
        model = modelgym.XGBModel(learning_task=par1)
        k = len(model.default_params)
        if model.learning_task == "classification":
            model.default_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif model.learning_task == "regression":
            model.default_params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        model.default_params['max_depth'] = int(model.default_params['max_depth'])
        assert test_params.__contains__(model.learning_task)
        assert len(model.default_params) != k + 3
    return 0


def test_convert_to_dataset(read_data):
    X, y, weights = read_data
    assert xgboost.DMatrix(X, y, None) == modelgym.XGBModel.convert_to_dataset(X, y, None)


def test_fit(read_data):
    X, y, weights = read_data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    params = def_params
    #result = xgboost.train(dtrain=dtrain, params=params)
    #assert result.score.call_count == 1
    return 0


def test_predict():
    return 0


@pytest.fixture
def read_data():
    with open("../data/XY2d.pickle", 'rb') as fh:
        X, y = pickle.load(fh, encoding='bytes')
    index = np.arange(X.shape[0])
    nrows = X.shape[0]
    weights = np.ones(nrows)  # uh, well.j..
    index_perm = np.random.permutation(index)
    return X[index_perm[:nrows]], y[index_perm[:nrows]], weights


@pytest.fixture
def preprocess(read_data):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(read_data, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    yield dtrain


def test_XGBModel():
    model = modelgym.XGBModel(learning_task="classification")
    param = def_params
    if model.learning_task == "classification":
        param.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
    elif model.learning_task == "regression":
        param.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
    param['max_depth'] = int(param['max_depth'])

    assert model.default_params == param
