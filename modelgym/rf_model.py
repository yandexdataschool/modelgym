from catboost import CatBoost
from sklearn.ensemble import RandomForestClassifier as rfc
from modelgym.model import Model
from hyperopt import hp, fmin, space_eval, tpe, STATUS_OK, Trials
from modelgym.XYCDataset import XYCDataset as xycd
from hyperopt.mongoexp import MongoTrials
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score


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
        # if self.learning_task == "classification":
        params.update({'verbose': 0})
        params['max_depth'] = int(params['max_depth'])
        return params

    def convert_to_dataset(self, data, label, cat_cols=None):
        ab = xycd(data, label, cat_cols)
        return ab

    def fit(self, params, dtrain, dtest, n_estimators):
        rf = rfc()
        bst = rf.fit(dtrain.X, dtrain.y)
        res = rf.predict(dtest.X)
        # print(res)
        return bst, res

    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest.X)
        return preds
