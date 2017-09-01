import pytest
from sklearn.metrics import roc_auc_score

import modelgym
from modelgym.gp_trainer import GPTrainer
from modelgym.util import TASK_CLASSIFICATION, split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
N_PROBES = 10
N_ESTIMATORS = 100
PARAMS_TO_TEST = ['eval_time', 'status', 'params']
FIT_TEST = (PARAMS_TO_TEST[:])
FIT_TEST.extend(['bst', 'n_estimators'])
CV_FIT = (PARAMS_TO_TEST[:])
CV_FIT.extend(['gp_eval_num', 'best_n_estimators'])
MAX_ROC_AUC_SCORE = 1.0


@pytest.mark.usefixtures("preprocess_data")
def test_crossval_fit_eval(preprocess_data):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    trainer = GPTrainer(gp_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    res = trainer.crossval_optimize_params(model=model, cv_pairs=cv_pairs)

    assert res['loss'] <= MAX_ROC_AUC_SCORE
    print("assert passed")

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
    res.pop('loss')
    res = model.preprocess_params(res)

    print("FITTING BEST PARAMS")
    bst, evals_result = model.fit(params=res, dtrain=_dtrain, dtest=_dtest, n_estimators=res['n_estimators'])
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.usefixtures("preprocess_data")
def test_fit_eval(preprocess_data):
    # TODO: contains all params? roc_auc
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    trainer = GPTrainer(gp_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    res = trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)

    assert res['loss'] <= MAX_ROC_AUC_SCORE
    print("assert passed")

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)
    res.pop('loss')
    res = model.preprocess_params(res)
    n_estimators = res['n_estimators']

    print("FITTING BEST PARAMS")
    bst, evals_result = model.fit(params=res, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.usefixtures("preprocess_data")
def test_crossval_optimize_params(preprocess_data):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    trainer = GPTrainer(gp_evals=N_PROBES, n_estimators=N_ESTIMATORS)

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)
    res = trainer.crossval_optimize_params(model=model, cv_pairs=cv_pairs)

    res.pop('loss')
    res = model.preprocess_params(res)
    n_estimators = res['n_estimators']

    bst, evals_result = model.fit(params=res, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)

    assert roc_auc <= MAX_ROC_AUC_SCORE
