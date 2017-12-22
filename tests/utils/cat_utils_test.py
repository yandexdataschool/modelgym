import pytest
import numpy as np
from modelgym.utils import XYCDataset, preprocess_cat_cols, cat_preprocess_cv

from sklearn.datasets import make_classification, make_regression


def test_cat_preprocess_cv():
    X, y = make_classification(n_samples=200, n_features=20,
                            n_informative=10, n_classes=2)
    dataset = XYCDataset(X, y)
    cv = dataset.cv_split(4)
    cv = cat_preprocess_cv(cv, one_hot_max_size=5)

    assert len(cv) == 4
    assert len(cv[0][0].cat_cols) == 0 and len(cv[0][1].cat_cols) == 0


def test_model_cat_preprocess():
    objects = 4
    features = 5

    cat_cols = [1, 2]

    y = np.arange(objects)

    make_set_X = lambda: \
        np.arange(objects * features).reshape((objects, features)).astype(float)

    X = make_set_X()
    R = preprocess_cat_cols(X, y, one_hot_max_size=1)
    # check that nothing changes
    assert np.array_equal(R, X)

    X = make_set_X()
    R = preprocess_cat_cols(X, y, cat_cols, one_hot_max_size=1)
    # check that one-hots aren't created
    assert R.shape == (4, 5)

    X = make_set_X()
    R  = preprocess_cat_cols(X, y, cat_cols, one_hot_max_size=4)
    # check 2 columns are transformed to one-hots
    assert R.shape == (4, 3 + 4 * 2)

    # check that one-hots contain only {0,1}
    assert set(np.unique(R[:, -8:].reshape(-1))) == set([0,1])


    A, B  = preprocess_cat_cols(X, y, cat_cols, X_test=X, one_hot_max_size=1)

    # check that two good args are returned if X_test is specified
    assert A.shape == X.shape
    assert B.shape == X.shape


def test_cat_preprocess_exceptions():
    X = np.array([['0', 'a'], ['0', 'b'], ['1', 'b'], ['1', 'c']])
    y = [0,0,1,1]

    # check that numeric strs '0'...'9' can be transformed
    try:
        A = preprocess_cat_cols(X, y, cat_cols=[0,1], one_hot_max_size=2)
    except ValueError or TypeError:
        assert False


    # check that other strings can't be transformed
    X = np.array([['0', 'a'], ['0', 'b'], ['1', 'b'], ['1', 'c']])
    isExc = False

    try:
        A = preprocess_cat_cols(X, y, cat_cols=[0,1], one_hot_max_size=3)
    except ValueError or TypeError:
        isExc = True

    assert isExc
