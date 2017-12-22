from modelgym.models.learning_task import LearningTask
from collections import defaultdict

from sklearn import preprocessing

import numpy as np

class CatCounter:
    """Categorical counter transformer class which calculates
    mean value of target for each unique label
    on prefix of random transposition of samples (like in catboost)
    """
    def __init__(self, learning_task, sort_values=None, seed=0):
        """
        Args:
            learning_task (LearningTask): type of learning task
            sort_values (None or numpy.ndarray): random transposition of indices
            seed (int): random seed
        """
        self.learning_task = learning_task
        self.sort_values = sort_values
        self.seed = seed
        self.sum_dicts = defaultdict(lambda: defaultdict(float))
        self.count_dicts = defaultdict(lambda: defaultdict(float))

    def update(self, value, col, key):
        self.sum_dicts[col][key] += value
        self.count_dicts[col][key] += 1

    def counter(self, key, col):
        num, den = self.sum_dicts[col][key], self.count_dicts[col][key]
        if self.learning_task == LearningTask.CLASSIFICATION:
            return (num + 1.) / (den + 2.)
        elif self.learning_task == LearningTask.REGRESSION:
            return num / den if den > 0 else 0
        else:
            raise ValueError('Task type must be classification or regression')

    def fit(self, X, y):
        self.sum_dicts = defaultdict(lambda: defaultdict(float))
        self.count_dicts = defaultdict(lambda: defaultdict(float))

        if self.sort_values is None:
            indices = np.arange(X.shape[0])
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        else:
            indices = np.argsort(self.sort_values)

        results = [np.zeros((X.shape[0], 0))]
        for col in range(X.shape[1]):
            result = np.zeros(X.shape[0])
            for index in indices:
                key = X[index, col]
                result[index] = self.counter(key, col)
                self.update(y[index], col, key)
            results.append(result.reshape(-1, 1))

        return np.concatenate(results, axis=1)

    def transform(self, X):
        results = [np.zeros((X.shape[0], 0))]
        for col in range(X.shape[1]):
            result = np.zeros(X.shape[0])
            for index in range(X.shape[0]):
                key = X[index, col]
                result[index] = self.counter(key, col)
            results.append(result.reshape(-1, 1))
        return np.concatenate(results, axis=1)


def cat_preprocess_cv(cv_pairs, one_hot_max_size=1,
        learning_task=LearningTask.CLASSIFICATION):
    """Default categorial columns preprocessing for each train-test split in cv

    Args:
        cv_pairs (list of tuples of XYCDataset):
        one_hot_max_size(int): max unique labels for one-hot-encoding
        learning_task (LearningTask): a type of learning task
    Returns:
        list of tuples of XYCDataset: cross validation folds
    """
    cv_prepared = []

    for dtrain, dtest in cv_pairs:
        preprocess_cat_cols(dtrain.X, dtrain.y,
            dtrain.cat_cols, dtest.X, one_hot_max_size, learning_task)
        dtrain.cat_cols = dtest.cat_cols = []
        cv_prepared.append((dtrain, dtest))

    return cv_prepared


def preprocess_cat_cols(X_train, y_train, cat_cols=[], X_test=None,
                one_hot_max_size=1, learning_task=LearningTask.CLASSIFICATION):
    """Preprocess categorial columns(cat_cols) in X_train
    and X_test(if specified) with cat-counting(the same as in catboost)
    or with one-hot-encoding,
    depends on number of unique labels(one_hot_max_size)

    Args:
        X_train (numpy.ndarray): train dataset
        y_train (numpy.ndarray): train labels
        cat_cols (list of columns indices): categorical columns
        X_test (None or numpy.ndarray): test dataset
        one_hot_max_size(int): max unique labels for one-hot-encoding
        learning_task (LearningTask): a type of learning task
    Returns:
        numpy.ndarray(, numpy.ndarray): transformed train and test datasets or
                                    only train, depends on X_test is None or not
    """
    one_hot_cols = [col for col in cat_cols
        if len(np.unique(X_train[:, col])) <= one_hot_max_size]

    cat_count_cols = list(set(cat_cols) - set(one_hot_cols))

    preprocess_counter_cols(X_train, y_train, cat_count_cols,
            X_test, learning_task=learning_task)

    X_train, X_test =  preprocess_one_hot_cols(X_train, one_hot_cols, X_test)

    if X_test is None:
        return X_train
    else:
        return X_train, X_test



def preprocess_counter_cols(X_train, y_train, cat_cols=None, X_test=None,
                        cc=None, counters_sort_col=None,
                        learning_task=LearningTask.CLASSIFICATION):
    """Transform columns(cat_cols) in X_train
    and X_test(if specified) with cat-counting(the same as in catboost)

    Args:
        X_train (numpy.ndarray): train dataset
        y_train (numpy.ndarray): train labels
        cat_cols (None or list of columns indices): categorical columns
        X_test (None or numpy.ndarray): test dataset
        cc (None or CatCounter): cat-counter fitted object
        counters_sort_col (None or numpy.ndarray): a prior order for
                                                        sorting samples
        learning_task (LearningTask): a type of learning task
    Returns:
        CatCounter: cat-counter fitted object
    """
    if cat_cols is None or len(cat_cols) == 0:
        return cc
    if cc is None:
       sort_values = None if counters_sort_col is None \
                                        else X_train[:, counters_sort_col]
       cc = CatCounter(learning_task, sort_values)
       X_train[:,cat_cols] = cc.fit(X_train[:,cat_cols], y_train)
    else:
       X_train[:,cat_cols] = cc.transform(X_train[:,cat_cols])
    if not X_test is None:
       X_test[:,cat_cols] = cc.transform(X_test[:,cat_cols])
    return cc


def preprocess_one_hot_cols(X_train, cat_cols=None, X_test=None):
    """Change columns(cat_cols) in X_train
    and X_test(if specified) into one-hot-encoding columns
    (always stacked to the right of the matrix).
    Data in cat_cols must have float or int type

    Args:
        X_train (numpy.ndarray): train dataset
        cat_cols (None or list of columns indices): categorical columns
        X_test (None or numpy.ndarray): test dataset
    Returns:
        numpy.ndarray, numpy.ndarray: transformed train and test datasets
    """
    add_one_hot = lambda X_old, X_one_hot: np.concatenate(
                            (np.delete(X_old, cat_cols, 1), X_one_hot), 1)

    if cat_cols is None or len(cat_cols) == 0:
        return X_train, X_test

    enc = preprocessing.OneHotEncoder(sparse=False)
    X_train = add_one_hot(X_train, enc.fit_transform(X_train[:, cat_cols]))

    if X_test is not None:
        X_test =  add_one_hot(X_test, enc.transform(X_test[:, cat_cols]))

    return X_train, X_test
