import numpy as np

from modelgym.models import Model, LearningTask
from hyperopt import hp
from hyperopt.pyll.base import scope

import copy
import os

from modelgym.models import XGBClassifier, LGBMClassifier, \
                            XGBRegressor, LGBMRegressor
from modelgym.utils import XYCDataset


class EnsembleClassifier(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict): parameters for model.
        """

        if params is None:
            params = {}

        self.params = params
        if 'models' not in params:
            raise ValueError('no models given')
        self.models = list(self.params['models'])

        self.weights = np.zeros(len(self.models))
        for i in range(len(self.models)):
            if 'weight_{}'.format(i) not in params:
                raise ValueError('weight_{} not given'.format(i))
            else:
                self.weights[i] = self.params['weight_{}'.format(i)]
        self.weights /= self.weights.sum()  # normalize

    def fit(self, dataset, weights=None, **kwargs):
        """
        Args:
            dataset (XYCDataset): train
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
            eval_dataset: same as dataset
            kwargs: CatBoost.Pool kwargs if eval_dataset == None or
                ``{'train': train_kwargs, 'eval': eval_kwargs}`` otherwise
        Return:
            self
        """
        params = kwargs if kwargs else self.params.get('fit_kwargs', {})
        for i, model in enumerate(self.models):
            self.models[i] = model.fit(dataset, weights=weights, **params)
        return self

    def save_snapshot(self, filename):
        """
        Args:
            filename: prefix for models' files
        Return:
            serializable internal model state snapshot.
        """
        state = {}
        state['params'] = self.params
        params = copy.copy(self.params)
        del params['models']
        np.save(filename + '_params', params)

        for i, model in enumerate(self.models):
            assert model, "model is not fitted"
            state['model_i'] = model.save_model(filename + '_{}'.format(i))
        return state

    @staticmethod
    def load_from_snapshot(self, filename, models):
        """
        Args:
            filename:  prefix for models' files
        Return:
            EnsembleClassifier
        """
        params = np.load(filename + '_params')

        loaded_models = []
        for i, model in enumerate(models):
            loaded_models.append(model.load_from_snapshot(filename + '_{}'.format(i)))
        params['models'] = loaded_models
        return EnsembleClassifier(params)

    @staticmethod
    def get_one_hot(targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    def predict(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        params = kwargs if kwargs else self.params.get('predict_kwargs', {})

        num_classes = self.params.get('num_class', 2)
        pred = np.zeros((dataset.X.shape[0], num_classes))
        for i, model in enumerate(self.models):
            pred += self.weights[i] * self.get_one_hot(
                model.predict(dataset, **params),
                num_classes
            )
        return np.argmax(pred, axis=1)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True  # any(x.is_possible_predict_proba() for x in self.models)

    def predict_proba(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, n_classes)
        """
        assert self.is_possible_predict_proba(), "Model cannot predict probability distribution"
        params = kwargs if kwargs else self.params.get('predict_proba_kwargs', {})
        num_classes = self.params.get('num_class', 2)

        pred = np.zeros((dataset.X.shape[0], num_classes))
        for i, model in enumerate(self.models):
            if model.is_possible_predict_proba():
                pred += self.weights[i] * self.get_one_hot(
                    model.predict(dataset, **params),
                    num_classes
                )
            else:
                pred += self.weights[i] * model.predict_proba(dataset, **params)

        if num_classes == 2:
            return pred[:, 1]
        return pred

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        return {
            'weight_0': hp.uniform('weight_0', 0, 1),
            'weight_1': hp.uniform('weight_1', 0, 1),
            'models': hp.choice('models', [[XGBClassifier(), LGBMClassifier()]])
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION


class EnsembleRegressor(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict): parameters for model
        """

        if params is None:
            params = {}

        self.params = params
        if 'models' not in params:
            raise ValueError('no models given')
        self.models = list(self.params['models'])

        self.weights = np.zeros(len(self.models))
        for i in range(len(self.models)):
            if 'weight_{}'.format(i) not in params:
                raise ValueError('weight_{} not given'.format(i))
            else:
                self.weights[i] = self.params['weight_{}'.format(i)]
        self.weights /= self.weights.sum()  # normalize

    def fit(self, dataset, weights=None, **kwargs):
        """
        Args:
            dataset (XYCDataset): train
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
            eval_dataset: same as dataset
            kwargs: CatBoost.Pool kwargs if eval_dataset == None or
                ``{'train': train_kwargs, 'eval': eval_kwargs}`` otherwise
        Return:
            self
        """
        params = kwargs if kwargs else self.params.get('fit_kwargs', {})
        for i, model in enumerate(self.models):
            self.models[i] = model.fit(dataset, weights=weights, **params)
        return self

    def save_snapshot(self, filename):
        """
        Args:
            filename: prefix for models' files
        Return:
            serializable internal model state snapshot.
        """
        state = {}
        state['params'] = self.params
        params = copy.copy(self.params)
        del params['models']
        np.save(filename + '_params', params)

        for i, model in enumerate(self.models):
            assert model, "model is not fitted"
            state['model_i'] = model.save_model(filename + '_{}'.format(i))
        return state

    @staticmethod
    def load_from_snapshot(self, filename, models):
        """
        Args:
            filename:  prefix for models' files
        Return:
            EnsembleClassifier
        """
        params = np.load(filename + '_params')

        loaded_models = []
        for i, model in enumerate(models):
            loaded_models.append(model.load_from_snapshot(filename + '_{}'.format(i)))
        params['models'] = loaded_models
        return EnsembleRegressor(params)

    def predict(self, dataset, **kwargs):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            kwargs: CatBoost.Pool kwargs
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        params = kwargs if kwargs else self.params.get('predict_kwargs', {})

        pred = np.zeros(dataset.X.shape[0])
        for i, model in enumerate(self.models):
            pred += self.weights[i] * model.predict(dataset, **kwargs)
        return pred

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False

    def predict_proba(self, dataset, **kwargs):
        """
        Regressor can't predict proba
        """
        raise ValueError("Regressor can't predict proba")

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """
        return {
            'weight_0': hp.uniform('weight_0', 0, 1),
            'weight_1': hp.uniform('weight_1', 0, 1),
            'models': hp.choice('models', [[XGBRegressor(), LGBMRegressor()]])
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.REGRESSION
