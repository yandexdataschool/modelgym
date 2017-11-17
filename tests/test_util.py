import numpy as np
import pytest
from hyperopt import hp
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
import sklearn.linear_model

import modelgym
from modelgym.model import TASK_CLASSIFICATION
from modelgym.util import split_and_preprocess, hyperopt2skopt_space, compare_models_different, XYCDataset
from modelgym import compare_auc_delong_xu

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
    fc = modelgym.RFModel(learning_task=TASK_CLASSIFICATION)
    print(hyperopt2skopt_space(fc.space))
    hyperopt_space = [{'a': hp.uniform('a', 0, 10)},
                      {'max_depth': hp.quniform('max_depth', 2, 10, 1)},
                      {'min_child_weight': hp.loguniform('min_child_weight', -16, 5)},
                      {'criterion': hp.choice('criterion', ["gini", "entropy"])},
                      {'max_features': hp.choice('max_features', range(1, 5))},
                      {'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1)},
                      {'abc': hp.quniform('abc', 0, 1)}
                      ]
    skopt_space = [Real(0, 10),
                   Integer(2, 10),
                   Real(np.exp(-16), np.exp(5)),
                   ["gini", "entropy"],
                   [1, 2, 3, 4],
                   Integer(int(np.exp(0)), int(np.exp(7))),
                   Integer(0, 1)
                   ]
    length = len(hyperopt_space)
    assert length == len(skopt_space)
    i = 1
    for p, r in zip(hyperopt_space, skopt_space):
        assert next(iter(modelgym.util.hyperopt2skopt_space(p).values())) == r
        print("[{0}/{1}] mini-test passed".format(i, length))
        i += 1


def test_variance():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target == 1, test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    y_test = y_test.astype(int)
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions)
    np.testing.assert_allclose(true_auc, auc)
    np.testing.assert_allclose(0.0014569635512, variance)


def test_compare_models():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target == 1, test_size=0.8, random_state=42)
    model = sklearn.linear_model.LogisticRegression()
    model.fit(x_train, y_train)

    y_test = y_test.astype(int)
    dtest = XYCDataset(x_test, y_test, [])
    assert compare_models_different(model, model, dtest)[0] == False

    second_model = sklearn.ensemble.RandomForestClassifier()
    second_model.fit(x_train, y_train)

    assert compare_models_different(model, second_model, dtest)[0] == True
