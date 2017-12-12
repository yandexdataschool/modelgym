import xgboost as xgb
import numpy as np

from modelgym.models import Model
from modelgym.utils import DistributionWrapper
import scipy.stats as sts


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
            'eta':               DistributionWrapper(sts.uniform, {"loc": 0, "scale": 0.1}),
            'max_depth':         DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'subsample':         DistributionWrapper(sts.uniform, {"loc": 0.5, "scale": 1}),
            'colsample_bytree':  DistributionWrapper(sts.uniform, {"loc": 0.5, "scale": 1}),
            'colsample_bylevel': DistributionWrapper(sts.uniform, {"loc": 0.5, "scale": 1}),
            'min_child_weight':  DistributionWrapper(sts.uniform, {"loc": 0, "scale": 0.1}),
             'gamma':            DistributionWrapper(sts.uniform, {"loc": 0, "scale": 1}),
            'lambdax':           DistributionWrapper(sts.uniform, {"loc": 0, "scale": 1}),
            'alpha':             DistributionWrapper(sts.uniform, {"loc": 0, "scale": 1}),
        }


class LGBMRegressor(Model):
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
          'learning_rate': DistributionWrapper(sts.uniform, {"loc": 0, "scale": 0.1}),
          'num_leaves': DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
          'feature_fraction': DistributionWrapper(sts.uniform, {"loc": 0.5, "scale": 1}),
          'bagging_fraction': DistributionWrapper(sts.uniform, {"loc": 0.5, "scale": 1}),
          'min_data_in_leaf': DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
          'min_sum_hessian_in_leaf': DistributionWrapper(sts.uniform, {"loc": 0, "scale": 0.1}),
          'lambda_l1': DistributionWrapper(sts.uniform, {"loc": 0, "scale": 1}),
          'lambda_l2': DistributionWrapper(sts.uniform, {"loc": 0, "scale": 1}),
        }
