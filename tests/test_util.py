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
    s1 = {'a': hp.uniform('a', 0, 10)}
    s1 = modelgym.util.hyperopt2skopt_space(s1)
    s2 = Real(0, 10)
    s3 = {'max_depth': hp.quniform('max_depth', 2, 10, 1)}
    s3 = modelgym.util.hyperopt2skopt_space(s3)
    s4 = Integer(2, 10)
    s5 = {'min_child_weight': hp.loguniform('min_child_weight', -16, 5)}
    s5 = modelgym.util.hyperopt2skopt_space(s5)
    s6 = Real(np.exp(-16), np.exp(5))
    s7 = Integer(np.exp(-16), np.exp(5))

    # TODO: more tests
    assert s1.get('a') == s2
    assert s3.get('max_depth') == s4
    assert s5.get('min_child_weight') == s6
    assert s7 != s6
