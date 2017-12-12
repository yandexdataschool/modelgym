import xgboost as xgb
import numpy as np

from modelgym.models import Model
import hyperopt as hp

class XGBClassifier(Model):
    def __init__(self, params=None):
        """
        :param params (dict or None): parameters for model.
                             If None default params are fetched.
        :param learning_task (str): set type of task(classification, regression, ...)
        """
        self.params = {'objective': 'binary:logistic', 'eval_metric': 'logloss',}
        self.params.update(params)
        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, dataset, weights=None):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :param y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        :param weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        :return: self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def save_snapshot(self, filename):
        """
        :return: serializable internal model state snapshot.

        """
        assert self.model, "model is not fitted"
        return self.model.save_model(filename)

    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        booster = xgb.Booster()
        booster.load_model(filename)
        return booster

    def predict(self, dataset):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :return: np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return np.argmax(self.model.predict(dataset.X), axis=1)

    def is_possible_predict_proba(self):
        """
        :return: bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :return: np.array, shape (n_samples, n_classes)
        """
        return self.model.predict(dataset.X)

    @staticmethod
    def get_default_parameter_space():
        """
        :return: dict of DistributionWrappers
        """

        return {
            'eta':               hp.loguniform('eta', -7, 0),
            'max_depth':         hp.quniform('max_depth', 2, 10, 1),
            'subsample':         hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight':  hp.loguniform('min_child_weight', -16, 5),
            'gamma':             hp.loguniform('gamma', -16, 2),
            'lambdax':           hp.loguniform('lambdax', -16, 2),
            'alpha':             hp.loguniform('alpha', -16, 2)
        }


class XGBRegressor(Model):
    def __init__(self, params=None):
        """
        :param params (dict or None): parameters for model.
                             If None default params are fetched.
        :param learning_task (str): set type of task(classification, regression, ...)
        """
        self.params = {'objective': 'reg:linear', 'eval_metric': 'rmse'}
        self.params.update(params)
        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)

    def fit(self, dataset, weights=None):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :param y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        :param weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        :return: self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def save_snapshot(self, filename):
        """
        :return: serializable internal model state snapshot.

        """
        assert self.model, "model is not fitted"
        return self.model.save_model(filename)

    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        booster = xgb.Booster()
        booster.load_model(filename)
        return booster

    def predict(self, dataset):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :return: np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        :return: bool, whether model can predict proba
        """
        return False

    def predict_proba(self, dataset):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :return: np.array, shape (n_samples, n_classes)
        """
        raise ValueError("Regressor can't predict proba")

    @staticmethod
    def get_default_parameter_space():
        """
        :return: dict of DistributionWrappers
        """

        return {
            'eta':               hp.loguniform('eta', -7, 0),
            'max_depth':         hp.quniform('max_depth', 2, 10, 1),
            'subsample':         hp.uniform('subsample', 0.5, 1),
            'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight':  hp.loguniform('min_child_weight', -16, 5),
            'gamma':             hp.loguniform('gamma', -16, 2),
            'lambdax':           hp.loguniform('lambdax', -16, 2),
            'alpha':             hp.loguniform('alpha', -16, 2)
        }
