import pytest
import numpy as np
import xgboost as xgb
import pandas as pd
from hyperopt import STATUS_OK
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

import modelgym
import pickle
from sklearn.cross_validation import train_test_split
import sklearn.datasets.data
from modelgym.util import split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
NROWS = 1000
N_PROBES = 2
N_ESTIMATORS = 100
TEST_PARAMS = ["classification", "range", "regression"]
APPROVED_PARAMS = ["classification", "regression"]


def test_preprocess_params():
    for par1 in TEST_PARAMS:
        # assert APPROVED_PARAMS.__contains__(par1)
        try:
            model = modelgym.XGBModel(learning_task=par1)
        except ValueError:
            print(par1, "not expected")
        k = len(model.default_params)
        model.preprocess_params(model.default_params)
        assert TEST_PARAMS.__contains__(model.learning_task)
        assert len(model.default_params) != k + 3
    return 0


def test_convert_to_dataset(preprocess):
    X_train, X_test, y_train, y_test = preprocess
    # print ('xtrain',X_train)
    # print('xtest',X_test)
    # print('ytrain',y_train)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    model = modelgym.XGBModel(learning_task=TEST_PARAMS[0])
    dexample = model.convert_to_dataset(data=X_train, label=y_train)
    # compare dtrain and dexample True
    # compare dtest and dexample False
    pass


def test_fit(read_cerndata):
    global roc_auc
    X, y, w = read_cerndata
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    from modelgym.trainer import Trainer
    from modelgym.util import TASK_CLASSIFICATION

    model = modelgym.XGBModel(TASK_CLASSIFICATION)

    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    res = trainer.crossval_fit_eval(model, cv_pairs)

    params = res['params']
    params = model.preprocess_params(params)
    n_estimators = params['n_estimators']

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    bst, evals_result = model.fit(params=params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}

    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    assert roc_auc > 0.5


def test_predict(read_cerndata):
    X, y, w = read_cerndata
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    from modelgym.trainer import Trainer
    from modelgym.util import TASK_CLASSIFICATION

    model = modelgym.XGBModel(TASK_CLASSIFICATION)

    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    res = trainer.crossval_fit_eval(model, cv_pairs)
    ans = trainer.fit_eval(model, dtrain, dtest, res['params'], res['best_n_estimators'],
                           custom_metric={'roc_auc': roc_auc_score})
    assert ans['roc_auc'] > 0.5


@pytest.fixture
def read_data():
    iris = sklearn.datasets.load_iris()
    return iris  # data and target


@pytest.fixture
def preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def read_cerndata():
    with open("data/XY2d.pickle", 'rb') as fh:
        X, y = pickle.load(fh, encoding='bytes')
    index = np.arange(X.shape[0])
    nrows = NROWS
    weights = np.ones(nrows)  # uh, well...
    index_perm = np.random.permutation(index)
    return X[index_perm[:nrows]], y[index_perm[:nrows]], weights
