from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier as rfc

from modelgym.XYCDataset import XYCDataset as xycd
from modelgym.model import Model


class RFModel(Model):
    def __init__(self, learning_task, compute_counters=False, counters_sort_col=None, holdout_size=0):
        Model.__init__(self, learning_task, 'RandomForestClassifier',
                       compute_counters, counters_sort_col, holdout_size)
        self.space = {
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features': hp.choice('max_features', range(1, 5)),
            'n_estimators': hp.choice('n_estimators', range(1, 20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
        }

        self.default_params = {
            'max_depth': 1,
            'max_features': 4,
            'n_estimators': 10,
            'criterion': "gini"
        }

        self.default_params = self.preprocess_params(self.default_params)

    def preprocess_params(self, params):
        params.update({'verbose': 0})
        params['max_depth'] = int(params['max_depth'])
        return params

    def convert_to_dataset(self, data, label, cat_cols=None):
        ab = xycd(data, label, cat_cols)
        return ab

    def fit(self, params, dtrain, dtest, n_estimators):
        rf = rfc(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'],
                 verbose=params['verbose'])
        bst = rf.fit(dtrain.X, dtrain.y)
        res = rf.predict(dtest.X)
        print("\n", res)
        return bst, res

    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest.X)
        return preds
