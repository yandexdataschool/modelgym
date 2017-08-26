import bson
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import modelgym
from modelgym.trainer import Trainer
from modelgym.util import TASK_CLASSIFICATION, split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
NROWS = 1000
N_PROBES = 2
N_ESTIMATORS = 100
PARAMS_TO_TEST = ['eval_time', 'status', 'params']
FIT_TEST = PARAMS_TO_TEST.copy()
FIT_TEST.extend(['bst', 'n_estimators'])
CV_FIT = PARAMS_TO_TEST.copy()
CV_FIT.extend(['hyperopt_eval_num', 'best_n_estimators'])


def test_crossval_fit_eval(preprocess_data):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)
    model = modelgym.XGBModel(TASK_CLASSIFICATION)

    res = trainer.crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=N_ESTIMATORS)

    for par1 in CV_FIT:
        assert par1 in res

    # TODO: why 0.5?
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
    print("ROC_AUC: ", roc_auc)
    assert roc_auc > 0.5


def test_fit_eval(preprocess_data):
    # TODO: contains all params? roc_auc
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)

    res = trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)

    for par1 in FIT_TEST:
        assert par1 in res

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
    print("ROC_AUC: ", roc_auc)
    assert roc_auc > 0.5


def test_crossval_optimize_params(preprocess_data):
    global roc_auc
    X_train, X_test, y_train, y_test = preprocess_data
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)

    model = modelgym.XGBModel(TASK_CLASSIFICATION)
    trainer = Trainer(hyperopt_evals=N_PROBES, n_estimators=N_ESTIMATORS)

    _dtrain = model.convert_to_dataset(dtrain.X, dtrain.y, dtrain.cat_cols)
    _dtest = model.convert_to_dataset(dtest.X, dtest.y, dtest.cat_cols)

    trainer.fit_eval(model, dtrain=dtrain, dtest=dtest)
    trainer.crossval_fit_eval(model=model, cv_pairs=cv_pairs, n_estimators=N_ESTIMATORS)

    optimized = trainer.crossval_optimize_params(model=model, cv_pairs=cv_pairs)

    params = trainer.best_params
    n_estimators = optimized.get('best_n_estimators')

    bst, evals_result = model.fit(params=params, dtrain=_dtrain, dtest=_dtest, n_estimators=n_estimators)
    prediction = model.predict(bst=bst, dtest=_dtest, X_test=dtest.X)

    custom_metric = {'roc_auc': roc_auc_score}
    for metric_name, metric_func in custom_metric.items():
        score = metric_func(_dtest.get_label(), prediction, sample_weight=None)  # TODO weights
        roc_auc = score
    print("ROC_AUC: ", roc_auc)
    assert roc_auc > 0.5


@pytest.fixture(scope="session")
def read_titanic():
    print("Reading data from CSV...")
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')

    def simplify_ages(df):
        df.Age = df.Age.fillna(-0.5)
        bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
        group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        categories = pd.cut(df.Age, bins, labels=group_names)
        df.Age = categories
        return df

    def simplify_cabins(df):
        df.Cabin = df.Cabin.fillna('N')
        df.Cabin = df.Cabin.apply(lambda x: x[0])
        return df

    def simplify_fares(df):
        df.Fare = df.Fare.fillna(-0.5)
        bins = (-1, 0, 8, 15, 31, 1000)
        group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
        categories = pd.cut(df.Fare, bins, labels=group_names)
        df.Fare = categories
        return df

    def format_name(df):
        df['Lname'] = df.Name.apply(lambda x: (x.split(' ')[0]).split(',')[0])
        df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
        return df

    def drop_features(df):
        return df.drop(['Ticket', 'Name'], axis=1)

    def transform_features(df):
        df = simplify_ages(df)
        df = simplify_cabins(df)
        df = simplify_fares(df)
        df = format_name(df)
        df = drop_features(df)
        return df

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)

    MaxPassEmbarked = data_train.groupby('Embarked').count()['PassengerId']
    data_train.Embarked[data_train.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[
        0]

    from sklearn import preprocessing
    def encode_features(df_train, df_test):
        features = ['Fare', 'Cabin', 'Age', 'Sex', 'Embarked', 'Lname', 'NamePrefix']
        df_combined = pd.concat([df_train[features], df_test[features]])

        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(df_combined[feature])
            df_train[feature] = le.transform(df_train[feature])
            df_test[feature] = le.transform(df_test[feature])
        return df_train, df_test

    data_train, data_test = encode_features(data_train, data_test)
    return data_train, data_test


@pytest.fixture(scope="session")
def preprocess_data(read_titanic):
    print("Processing data from CSV...")
    data_train, data_test = read_titanic
    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']
    X_all = X_all.values
    y_all = y_all.values

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test
