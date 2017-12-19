import pytest
import numpy as np
from modelgym.utils import XYCDataset
from modelgym.cat_utils import preprocess_cat_cols


def test_model_cat_preprocess():
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
