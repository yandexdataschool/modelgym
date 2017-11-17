import pytest
from sklearn.metrics import roc_auc_score

import modelgym
from modelgym import TPETrainer, RandomTrainer, GPTrainer, ForestTrainer
from modelgym.util import TASK_CLASSIFICATION, split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
N_PROBES = 10
N_ESTIMATORS = 100
PARAMS_TO_TEST = ['eval_time', 'status', 'params']
MAX_ROC_AUC_SCORE = 1.0
TRAINER_CLASS = [TPETrainer, RandomTrainer, GPTrainer, ForestTrainer]
MODEL_CLASS = [modelgym.XGBModel, modelgym.LGBModel, modelgym.RFModel]


@pytest.mark.parametrize("model_class", MODEL_CLASS)
@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
@pytest.mark.usefixtures("preprocess_data")
def test_crossval_fit_eval(preprocess_data, trainer_class, model_class):
    global roc_auc, res
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    trainer = trainer_class(opt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    model = model_class(TASK_CLASSIFICATION)
    flag = False
    for i in range(0, 2):
        if (flag):
            res = trainer.crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=N_ESTIMATORS,
                                            params=model.default_params)
            flag = True
        else:
            res = trainer.crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=N_ESTIMATORS)
    if isinstance(res, dict):
        loss = res['loss']
        params = res['params']
        params = model.preprocess_params(params)
    else:
        loss = res
        params = model.default_params
    assert loss <= MAX_ROC_AUC_SCORE
    n_estimators = N_ESTIMATORS

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    print("FITTING BEST PARAMS")
    bst, evals_result = model.fit(params=params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.parametrize("model_class", MODEL_CLASS)
@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
@pytest.mark.usefixtures("preprocess_data")
def test_fit_eval(preprocess_data, trainer_class, model_class):
    # TODO: contains all params? roc_auc
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    model = model_class(TASK_CLASSIFICATION)
    trainer = trainer_class(opt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    res = trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)

    loss = res['loss']
    params = res['params']
    params = model.preprocess_params(params)
    res.pop('loss')
    assert loss <= MAX_ROC_AUC_SCORE
    n_estimators = N_ESTIMATORS

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    bst, evals_result = model.fit(params=params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc <= MAX_ROC_AUC_SCORE


@pytest.mark.parametrize("model_class", MODEL_CLASS)
@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
@pytest.mark.usefixtures("preprocess_data")
def test_crossval_optimize_params(preprocess_data, trainer_class, model_class):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    model = model_class(TASK_CLASSIFICATION)
    trainer = trainer_class(opt_evals=N_PROBES, n_estimators=N_ESTIMATORS)

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    # trainer.fit_eval(model=model, dtrain=dtrain, dtest=dtest)
    optimized = trainer.crossval_optimize_params(model=model, cv_pairs=cv_pairs)

    optimized.pop('loss')
    params = optimized['params']
    params = model.preprocess_params(params)
    n_estimators = N_ESTIMATORS

    bst, evals_result = model.fit(params=params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)

    assert roc_auc <= MAX_ROC_AUC_SCORE
