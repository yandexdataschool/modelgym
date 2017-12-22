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


def test_model_cat_preprocess_numeric():
    objects = 4
    features = 5

    cat_cols = [1, 2]

    y = np.arange(objects)

    X = np.arange(objects * features).reshape((objects, features)).astype(float)

    A = preprocess_cat_cols(X, y, one_hot_max_size=1)

    print(A)

    assert np.array_equal(A, X)

    X = preprocess_cat_cols(X, y, cat_cols, one_hot_max_size=1)

    print(X)
    # check that one-hots aren't created
    assert X.shape == (4, 5)

    X = np.arange(objects * features).reshape((objects, features)).astype(float)

    print(X)

    X  = preprocess_cat_cols(X, y, cat_cols, one_hot_max_size=4)

    print(X)

    # check 2 columns are transformed to one-hots
    assert X.shape == (4, 3 + 4 * 2)

    # check that one-hots contain only {0,1}
    assert set(np.unique(X[:, -8:].reshape(-1))) == set([0,1])


    A, B  = preprocess_cat_cols(X, y, cat_cols, X_test=X, one_hot_max_size=1)

    # check that two good args are returned if X_test is specified
    assert A.shape == X.shape
    assert B.shape == X.shape



def test_model_cat_preprocess_str():

    X = np.array([['a', 'b'], ['b', 'b'], ['b', 'b']])

    y = [0, 0, 1]

    A = preprocess_cat_cols(X, y, one_hot_max_size=1)

    print(A)

    assert np.array_equal(A, X)

    A = preprocess_cat_cols(X, y, one_hot_max_size=5)

    print(A)

    assert np.array_equal(A, X)

    # assert False
