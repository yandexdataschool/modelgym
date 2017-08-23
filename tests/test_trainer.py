import pytest
from sklearn.model_selection import train_test_split

from modelgym.trainer import Trainer
import xgboost as xgb
import modelgym
import sklearn.datasets.data
import time
from hyperopt import fmin, Trials, STATUS_OK, STATUS_FAIL
from modelgym.util import TASK_CLASSIFICATION, split_and_preprocess

TEST_SIZE = 0.5
N_CV_SPLITS = 2
N_ESTIMATORS = 100
N_PROBES = 2


def test_fit_eval(preprocess):
    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    model_class = modelgym.XGBModel
    model = model_class(TASK_CLASSIFICATION)
    X_train, X_test, y_train, y_test = preprocess
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    res = trainer.fit_eval(model=model, dtrain=dtrain, dtest=dtest, n_estimators=N_ESTIMATORS)

    params = model.default_params
    n_estimators = N_ESTIMATORS
    params = model.preprocess_params(params)
    start_time = time.time()
    _dtrain = model.convert_to_dataset(dtrain.get, dtrain[1], dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
    bst, evals_result = model.fit(params, _dtrain, _dtest, n_estimators)
    eval_time = time.time() - start_time

    expected = {'loss': evals_result[-1], 'bst': bst, 'n_estimators': n_estimators,
                'eval_time': eval_time, 'status': STATUS_OK, 'params': params.copy()}
    assert expected == res


def test_crossval_fit_eval():
    # TODO: implememnt
    return 0


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
