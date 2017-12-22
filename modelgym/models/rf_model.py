import pickle

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier as rfc

from modelgym.models import Model, LearningTask
from modelgym.utils.dataset import XYCDataset
from hyperopt import hp


class RFClassifier(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}

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
        if params:
            self.params.update(params)

        self.n_estimators = self.params.pop('n_estimators', 1)
        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return XYCDataset(data, label, cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = rfc(n_estimators=self.n_estimators, max_depth=self.params['max_depth'],
                         criterion=self.params['criterion'], max_features=self.params['max_features'],
                         verbose=self.params['verbose']).fit(dtrain.X, dtrain.y)
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.

        """
        assert self.model is not None, "model is not fitted"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        new_model = RFClassifier(model.get_params())
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        return self.model.predict_proba(dataset.X)[:, 1]

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """

        return {
            'n_estimators':      hp.choice('n_estimators', range(5, 7)),
            'max_depth':         hp.choice('max_depth', range(1, 20)),
            'max_features':      hp.choice('max_features', range(1, 5)),
            'n_estimators':      hp.choice('n_estimators', range(1, 20)),
            'criterion':         hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
            'min_samples_leaf':  hp.quniform('min_samples_leaf', 1, 20, 1),
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION
