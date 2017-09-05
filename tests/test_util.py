import numpy as np
import pytest
from hyperopt import hp
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer

import modelgym
from modelgym.util import split_and_preprocess

TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
N_CV_SPLITS = 2


@pytest.mark.usefixtures("read_data")
def test_split_and_preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    # TODO: more tests
    assert len(X_test) <= TEST_SIZE * len(iris_data.data)
    assert len(X_train) <= TRAIN_SIZE * len(iris_data.data)
    assert len(y_test) <= TEST_SIZE * len(iris_data.data)
    assert len(y_train) <= TRAIN_SIZE * len(iris_data.data)
    assert (len(X_train) == len(y_train))
    assert (len(X_test) == len(y_test))


def test_hyperopt2skopt_space():
    hyperopt_space = [{'a': hp.uniform('a', 0, 10)}, {'max_depth': hp.quniform('max_depth', 2, 10, 1)},
                      {'min_child_weight': hp.loguniform('min_child_weight', -16, 5)},
                      {'criterion': hp.choice('criterion', ["gini", "entropy"])},
                      {'max_features': hp.choice('max_features', range(1, 5))}, {'abc': hp.uniform('abc', 0, 1)}
                      ]
    skopt_space = [Real(0, 10), Integer(2, 10), Real(np.exp(-16), np.exp(5)), ["gini", "entropy"], [1,2,3,4],Integer(0, 1)]
    for p, r in zip(hyperopt_space, skopt_space):
        assert next(iter(modelgym.util.hyperopt2skopt_space(p).values())) == r
