from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier as rfc

from modelgym.util import XYCDataset as xycd
from modelgym.model import Model, TASK_CLASSIFICATION


class RFModel(Model):
    def __init__(self, learning_task, compute_counters=False, counters_sort_col=None, holdout_size=0):
        Model.__init__(self, learning_task, 'RandomForestClassifier',
                       compute_counters, counters_sort_col, holdout_size)
        self.space = {
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features': hp.choice('max_features', range(1, 5)),
            'n_estimators': hp.choice('n_estimators', range(1, 20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
        }

        self.default_params = {
            'max_depth': 1,
            'max_features': 4,
            'n_estimators': 10,
            'criterion': "gini",
            'verbose': 0,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.,
            'min_impurity_split': 1e-7
        }

        self.default_params = self.preprocess_params(self.default_params)

    def preprocess_params(self, params):
        if not (self.learning_task == TASK_CLASSIFICATION):
            raise ValueError()
        params_ = params.copy()
        if (isinstance(params_, dict)):
            params_.update({'verbose': 0})
            if (params_.__contains__('max_depth')):
                params_['max_depth'] = int(params_['max_depth'])
        return params_

    def set_parameters(self, params, **kwargs):
        if (isinstance(params, list)):
            max_depth, max_features, n_estimators, criterion, min_samples_split, min_samples_leaf = \
                params[0], params[1], params[2], params[3], params[4], params[5]
            self.default_params.update(max_depth=max_depth,
                                       max_features=max_features,
                                       n_estimators=n_estimators,
                                       criterion=criterion,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf
                                       )
        else:
            self.default_params.update(kwargs)

    def convert_to_dataset(self, data, label, cat_cols=None):
        ab = xycd(data, label, cat_cols)
        return ab

    def fit(self, params, dtrain, dtest, n_estimators):
        rf = rfc(n_estimators=n_estimators, max_depth=params['max_depth'],
                 criterion=params['criterion'], max_features=params['max_features'], verbose=params['verbose']).fit(
            dtrain.X, dtrain.y)
        preds = rf.predict(dtest.X)
        return rf, preds

    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest.X)
        return preds
