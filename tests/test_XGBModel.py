import pytest
import string
import numpy as np
import random
import xgboost
import pandas as pd
import modelgym
import pickle
from sklearn.model_selection import train_test_split
from modelgym.util import split_and_preprocess

def test_preprocess_params():
    return 0

def test_convert_to_dataset(load_data):
    X,y, weights=load_data
    assert xgboost.DMatrix(X, y, None)==modelgym.XGBModel.convert_to_dataset(X,y,None)

@pytest.mark.fast_test
def test_fit():
    return 0

@pytest.mark.fast_test
def test_predict():
    return 0

def read_data(fname):
    with open(fname, 'rb') as fh:
        X, y = pickle.load(fh, encoding='bytes')
    index = np.arange(X.shape[0])
    nrows = X.shape[0]
    weights = np.ones(nrows)  # uh, well.j..
    index_perm = np.random.permutation(index)
    return X[index_perm[:nrows]], y[index_perm[:nrows]], weights

@pytest.fixture
def load_data(read_data):
    s="../data/XY2d.pickle"
    yield  read_data,s

@pytest.fixture
def preprocess(load_data):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(load_data, test_size=0.5)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=2)
    yield dtrain

def test_XGBModel():
    model = modelgym.XGBModel(learning_task="classification")
    param = {'base_score': 0.5,
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
    assert model.default_params == param
