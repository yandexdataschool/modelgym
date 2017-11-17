import os
import tempfile

import pytest
import skopt
import xgboost
from sklearn.metrics import roc_auc_score
import numpy as np
from skopt.space import Real, Integer, Space

import modelgym
from modelgym.trainer import Trainer
from modelgym.util import split_and_preprocess
from modelgym.model import TASK_CLASSIFICATION, TASK_REGRESSION

TEST_SIZE = 0.2
N_CV_SPLITS = 2
N_PROBES = 2
N_ESTIMATORS = 100
MAX_ROC_AUC_SCORE = 1.0
APPROVED_PARAMS = [TASK_CLASSIFICATION, TASK_REGRESSION]
TEST_PARAMS = (APPROVED_PARAMS[:])
TEST_PARAMS.append("range")
MODEL_CLASS = [modelgym.XGBModel, modelgym.LGBModel, modelgym.RFModel]


def test_preprocess_params():
    for par1 in TEST_PARAMS:
        try:
            model = modelgym.XGBModel(learning_task=par1)
        except ValueError:
            print(par1, "not expected")
        k = len(model.default_params)
        model.preprocess_params(model.default_params)
        assert TEST_PARAMS.__contains__(model.learning_task)
        assert len(model.default_params) != k + 3
    return 0


@pytest.mark.usefixtures("preprocess_data")
def test_convert_to_dataset(preprocess_data):
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
    _dexample = xgboost.DMatrix(data=dtrain.X, label=dtrain.y)
    assert _dtrain.num_row() == _dexample.num_row()
    assert _dtrain.num_col() == _dtest.num_col() == _dexample.num_col()
    assert _dtest.num_row() != _dexample.num_row()


@pytest.mark.usefixtures("preprocess_data")
@pytest.mark.parametrize("model_class", MODEL_CLASS)
def test_fit(preprocess_data, model_class):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = model_class(TASK_CLASSIFICATION)
    n_estimators = 10

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    bst, evals_result = model.fit(params=model.default_params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.usefixtures("preprocess_data")
def test_predict(preprocess_data):
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    trainer = Trainer(opt_evals=N_PROBES, n_estimators=N_ESTIMATORS)

    res = trainer.crossval_fit_eval(model, cv_pairs)
    ans = trainer.fit_eval(model, dtrain, dtest, res['params'], res['best_n_estimators'],
                           custom_metric={'roc_auc': roc_auc_score})
    roc_auc = ans['roc_auc']
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.parametrize("model_class", MODEL_CLASS)
@pytest.mark.parametrize('task', APPROVED_PARAMS)
def test_load_and_save(model_class, task):
    try:
        model1 = model_class(learning_task=task)  # model to save and then read
    except ValueError:
        print("can't initialize model: {} with task: {}".format(model_class, task))
        return
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        filepath = tmp.name
        model1.save_config(filepath)
        assert os.path.exists(filepath)
        model2 = model_class(learning_task=task)
        model2.load_config(filepath)
        dic1 = model1.__dict__
        dic2 = model2.__dict__
        # check all values match
        assert dic1.keys() == dic2.keys()
        params1 = dic1.get("space")
        params2 = dic2.get("space")
        if params1 != params2:
            print("\n")
            for param in params1:
                # print(str(params2.__getitem__(param)))
                assert str(params1.__getitem__(param)) == str(params2.__getitem__(param)), "exit3"
        else:
            print("params equal")
