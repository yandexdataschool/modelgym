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

class TestXGBModel():
    def test_preprocess_params():
        return 0

    def test_convert_to_dataset(load_data):
        dtrain=load_data
        assert xgboost.DMatrix(dtrain.X, dtrain.y, None)==modelgym.XGBModel.convert_to_dataset(dtrain.X,dtrain.y,None)

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
        weights = np.ones(nrows)  # uh, well...
        index_perm = np.random.permutation(index)
        return X[index_perm[:nrows]], y[index_perm[:nrows]], weights

    @pytest.fixture
    def load_data(read_data):
        X, y, weights = read_data("../data/XY2d.pickle")
        return X,y,weights

    @pytest.fixture
    def preprocess(load_data):
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(load_data, test_size=0.5)
        cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                         X_test.copy(), y_test,
                                                         cat_cols=[], n_splits=2)
        return dtrain