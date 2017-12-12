from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import log_loss, mean_squared_error

from modelgym.models import Model
from modelgym.utils import DistributionWrapper, XYCDataset as xycd
import scipy.stats as sts


class RFClassifier(Model):
    def __init__(self, params=None):
        """
        :param params (dict or None): parameters for model.
                             If None default params are fetched.
        :param learning_task (str): set type of task(classification, regression, ...)
        """
        self.params = {
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
        self.params.update(params)

        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def convert_to_dataset(self, data, label, cat_cols=None):
        return xycd(data, label, cat_cols)

    def fit(self, dataset, weights=None):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :param y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        :param weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        :return: self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = rfc(n_estimators=self.n_estimators, max_depth=self.params['max_depth'],
                         criterion=self.params['criterion'], max_features=self.params['max_features'],
                         verbose=self.params['verbose']).fit(dtrain.X, dtrain.y)
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
        return True

    def predict_proba(self, dataset):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :return: np.array, shape (n_samples, n_classes)
        """
        return self.model.predict_proba(dataset.X)[:, 1]

    @staticmethod
    def get_default_parameter_space():
        """
        :return: dict of DistributionWrappers
        """

        return {
            'max_depth':         DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'max_features':      DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'max_features':      DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'n_estimators':      DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'colsample_bylevel': DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'min_samples_split': DistributionWrapper(sts.randint, {"a": 1, "b": 7}),
            'min_samples_leaf':  DistributionWrapper(sts.randint, {"a": 1, "b": 7}),

        }

