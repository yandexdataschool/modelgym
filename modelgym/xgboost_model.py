import xgboost as xgb
from hyperopt import hp

from modelgym.model import Model, TASK_CLASSIFICATION


class XGBModel(Model):
    defspace = {
        'max_depth': (1, 5),
        'learning_rate': (10 ** -4, 10 ** -1),
        'n_estimators': (10, 200),
        'min_child_weight': (1, 20),
        'subsample': (0, 1),
        'colsample_bytree': (0.3, 1)
    }
    def __init__(self, learning_task, compute_counters=False, counters_sort_col=None, holdout_size=0):
        Model.__init__(self, learning_task, 'XGBoost',
                       compute_counters, counters_sort_col, holdout_size)
        self.space = {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
        }

        self.default_params = {'base_score': 0.5,
                               'colsample_bylevel': 1,
                               'colsample_bytree': 1,
                               'gamma': 0,
                               'learning_rate': 0.1,
                               'max_delta_step': 0,
                               'max_depth': 3,
                               'min_child_weight': 1,
                               'missing': None,
                               'n_estimators': 100,
                               'nthread': -1,
                               'reg_alpha': 0,
                               'reg_lambda': 1,
                               'scale_pos_weight': 1,
                               'seed': 0,
                               'subsample': 1}

        self.default_params = self.preprocess_params(self.default_params)

    def preprocess_params(self, params):
        if self.learning_task == "classification":
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif self.learning_task == "regression":
            params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        if (params.__contains__('max_depth')):
            params['max_depth'] = int(params['max_depth'])
        return params

    def set_params(self, **kwargs):
        for key in kwargs:
            print(key, "\t")
            dic = {key: kwargs.get(key)}
            print(dic, "\t")
            self.default_params.update(dic)

    def convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, params, dtrain, dtest, n_estimators):
        evals_result = {}
        if (not isinstance(params, dict)):
            attr = ['max_depth',
                    'learning_rate',
                    'n_estimators',
                    'min_child_weight',
                    'subsample',
                    'colsample_bytree']
            params = {k: v for k, v in zip(attr, params)}
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        print("\t", params)
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)
        results = evals_result['test']['rmse'] if self.learning_task == 'regression' \
            else evals_result['test']['logloss']
        print('\t FITTED')
        return bst, results

    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest)
        return preds
