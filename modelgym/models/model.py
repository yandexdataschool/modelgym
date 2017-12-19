from modelgym.models.learning_task import LearningTask
from modelgym.cat_utils import preprocess_cat_cols


class Model(object):
    """
    Model is a base class for a specific ML algorithm implementation factory,
    i.e. it defines algorithm-specific hyperparameter space and generic methods for model training & inference
    """

    def __init__(self, params=None):
        """
        :param params (dict or None): parameters for model.
        """
        raise NotImplementedError("Pure virtual class.")

    def _set_model(self, model):
        """
        sets new model
        :param model: internal model
        """
        raise NotImplementedError("Pure virtual class.")

    def fit(self, dataset, weights=None):
        """
        :param X (np.array, shape (n_samples, n_features)): the input data
        :param y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        :param weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        :return: self
        """
        raise NotImplementedError("Pure virtual class.")

    def save_snapshot(self, filename):
        """
        :return: serializable internal model state snapshot.

        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        raise NotImplementedError("Pure virtual class.")

    def predict(self, dataset):
        """
        :param dataset (modelgym.utils.XYCDataset): the input data,
            dataset.y may be None
        :return: np.array, shape (n_samples, ) -- predictions
        """
        raise NotImplementedError("Pure virtual class.")

    def is_possible_predict_proba(self):
        """
        :return: bool, whether model can predict proba
        """
        raise NotImplementedError("Pure virtual class.")

    def predict_proba(self, X):
        """
        :param dataset (np.array, shape (n_samples, n_features)): the input data
        :return np.array, shape (n_samples, n_classes): predicted probabilities
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def get_default_parameter_space():
        """
        :return dict from parameter name to hyperopt distribution: default
            parameter space
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def get_learning_task(self):
        """
        :return modelgym.models.LearningTask: task
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def cat_preprocess(cv_pairs, one_hot_max_size=1,
            learning_task=LearningTask.CLASSIFICATION):
        """default categorical features preprocessing
        :param cv_pairs list of tuples of 2 XYCDataset's: cross validation folds
            for preparation
        :return list of tuples of 2 XYCDataset's: cross validation folds
        """
        cv_prepared = []

        for dtrain, dtest in cv_pairs:
            preprocess_cat_cols(dtrain.X, dtrain.y,
                dtrain.cat_cols, dtest.X, one_hot_max_size, learning_task)
            cv_prepared.append((dtrain, dtest))

        return cv_prepared
