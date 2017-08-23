import pickle

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from modelgym.trainer import Trainer
import modelgym
import sklearn.datasets.data
from modelgym.util import TASK_CLASSIFICATION, split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
NROWS = 1000
N_PROBES = 2
N_ESTIMATORS = 100


def test_crossval_fit_eval(read_cerndata):
    global roc_auc
    X, y, w = read_cerndata
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=TEST_SIZE)
    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    res = trainer.crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=N_ESTIMATORS)
    # TODO: why 0.5? what is loss?
    # TODO: contains all params?
    assert res['loss'] < 0.5

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


def test_fit_eval(read_cerndata):
    # TODO: contains all params?
    global roc_auc
    X, y, w = read_cerndata
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    model = modelgym.XGBModel(TASK_CLASSIFICATION)

    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    res = trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)

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


def test_crossval_optimize_params():
    # TODO: implememnt
    return 0


@pytest.fixture
def read_data():
    iris = sklearn.datasets.load_iris()
    return iris  # data and target


@pytest.fixture
def preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
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
