import lightgbm as lgb
from hyperopt import hp
from modelgym.model import Model
import numpy as np


class LGBModel(Model):
    def __init__(self, learning_task, compute_counters=False, counters_sort_col=None, holdout_size=0):
        Model.__init__(self, learning_task, 'LightGBM',
                       compute_counters, counters_sort_col, holdout_size)

        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        }

        self.default_params = {  # 'learning_rate': 0.1,
            # 'num_leaves': 127, changed to 31
            # 'feature_fraction': 1.0,
            # 'bagging_fraction': 1.0, # not found
            # 'min_data_in_leaf': 100, # actually 20
            # 'min_sum_hessian_in_leaf': 10, changed to 1e-3
            'lambda_l1': 0,
            'lambda_l2': 0,
            'num_threads': 4}
        self.default_params = {'boosting_type': 'gbdt',
                               'colsample_bytree': 1,
                               'drop_rate': 0.1,
                               'is_unbalance': False,
                               'learning_rate': 0.1,
                               'max_bin': 255,
                               'min_data_in_leaf': 20,
                               'max_depth': -1,
                               'max_drop': 50,
                               'min_child_samples': 10,
                               'min_child_weight': 5,
                               'min_split_gain': 0,
                               'min_sum_hessian_in_leaf': 1e-3,
                               'lambda_l1': 0,
                               'lambda_l2': 0,
                               'n_estimators': 10,
                               'nthread': 4,
                               'num_threads': 4,
                               'num_leaves': 31,
                               'reg_alpha': 0,
                               'reg_lambda': 0,
                               'scale_pos_weight': 1,
                               'seed': 0,
                               'sigmoid': 1.0,
                               'skip_drop': 0.5,
                               'subsample': 1,
                               'subsample_for_bin': 50000,
                               'subsample_freq': 1,
                               'uniform_drop': False,
                               'xgboost_dart_mode': False}
        self.default_params = self.preprocess_params(self.default_params)

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task == 'classification':
            params_.update({'objective': 'binary', 'metric': 'binary_logloss',
                            'bagging_freq': 1, 'verbose': -1})
        elif self.learning_task == "regression":
            params_.update({'objective': 'mean_squared_error', 'metric': 'l2',
                            'bagging_freq': 1, 'verbose': -1})
        params_['num_leaves'] = max(int(params_['num_leaves']), 2)
        params_['min_data_in_leaf'] = int(params_['min_data_in_leaf'])
        return params_

    def convert_to_dataset(self, data, label, cat_cols=None):
        return lgb.Dataset(data, label)

    def fit(self, params, dtrain, dtest, n_estimators):
        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)

        results = np.power(evals_result['test']['l2'], 0.5) if self.learning_task == 'regression' \
            else evals_result['test']['binary_logloss']
        return bst, results

    def predict(self, bst, dtest, X_test):
        preds = bst.predict(X_test)
        return preds
