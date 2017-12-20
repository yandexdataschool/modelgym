from modelgym.models.learning_task import LearningTask


class Model(object):
    """
    Model is a base class for a specific ML algorithm implementation factory,
    i.e. it defines algorithm-specific hyperparameter space and generic methods for model training & inference
    """

    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
        """
        raise NotImplementedError("Pure virtual class.")

    def _set_model(self, model):
        """
        sets new model
        Args:
            model: internal model
        """
        raise NotImplementedError("Pure virtual class.")

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        raise NotImplementedError("Pure virtual class.")

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.

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
        Args:
            dataset (modelgym.utils.XYCDataset): the input data,
                dataset.y may be None
        Return:
            np.array, shape (n_samples, ): predictions
        """
        raise NotImplementedError("Pure virtual class.")

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        raise NotImplementedError("Pure virtual class.")

    def predict_proba(self, X):
        """
        Args:
            dataset (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes): predicted probabilities
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict from parameter name to hyperopt distribution: default
            parameter space
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def get_learning_task():
        """
        Return:
            modelgym.models.LearningTask: task
        """
        raise NotImplementedError("Pure virtual class.")
