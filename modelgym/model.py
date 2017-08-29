import os
import yaml
TASK_CLASSIFICATION = 'classification'
TASK_REGRESSION = 'regression'


class Model(object):
    def __init__(self, learning_task=TASK_CLASSIFICATION, bst_name=None,
                 compute_counters=True, counters_sort_col=None, holdout_size=0):
        self.learning_task, self.bst_name = learning_task, bst_name
        self.compute_counters = compute_counters
        self.holdout_size = holdout_size
        self.counters_sort_col = counters_sort_col
        self.default_params, self.best_params = None, None
        self.space = None
        if self.learning_task == TASK_CLASSIFICATION:
            self.metric = 'logloss'
        elif self.learning_task == TASK_REGRESSION:
            self.metric = 'rmse'
        else:
            raise ValueError('Task type must be "classification" or "regression"')

    def __iter__(self):
        yield 'learning_task', self.learning_task
        yield 'bst_name', self.bst_name
        yield 'compute_counters', self.compute_counters
        yield 'counters_sort_col', self.counters_sort_col
        yield 'default_params', self.default_params
        yield 'holdout_size', self.holdout_size
        yield 'metric', self.metric
        yield 'space', self.space

    def get_name(self):
        return self.bst_name  # TODO: rename

    def convert_to_dataset(self, data, label, cat_cols=None):
        raise NotImplementedError('Method convert_to_dataset is not implemented.')

    def preprocess_params(self, params):
        raise NotImplementedError('Method preprocess_params is not implemented.')

    def fit(self, params, dtrain, dtest, n_estimators):
        raise NotImplementedError('Method train is not implemented.')

    def predict(self, bst, dtest, X_test):
        raise NotImplementedError('Method predict is not implemented.')

    def load_config(self, filepath):
        if os.path.exists(filepath):
            with open(filepath) as f:
                dataMap = yaml.load(f)
                self.__dict__.update(dataMap)
        else:
            raise ValueError('Model {0} do not exist'.format(filepath))

    def save_config(self, filepath):
        with open(filepath, "w") as f:
            yaml.dump(self, f)
