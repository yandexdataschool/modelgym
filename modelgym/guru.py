from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import scipy.stats as ss


class Guru:
    """
    This class analyze data trying to find some issues.
    """
    _NOT_NUMERIC_KEY = 'not numeric'
    _NOT_VARIABLE_KEY = 'not variable'
    _TOO_RARE_KEY = 'too rare'
    _TOO_COMMON_KEY = 'too common'
    _SPARSE = 'sparse'
    _CATEGORIAL = 'categorial'
    _CLASS_DISBALANCE = 'class disbalance'
    _CORRELATION = 'correlation'
    _DEFAULT_CMAP = 'hot'

    _MESSAGE_DICT = {_SPARSE: 'Consider use hashing trick for ' +
                              'your sparse features, if you haven\'t ' +
                              'already. Following features are ' +
                              'supposed to be sparse: ',
                     _CORRELATION: 'There are several correlated features. ' +
                                   'Consider dimention reduction, ' +
                                   'for example you can use PCA. ' +
                                   'Following pairs of features are supposed to be' +
                                   ' correlated: ',
                     _CATEGORIAL: 'Some features are ' +
                                  'supposed to be categorial. Make sure ' +
                                  'that all categorial features are in ' +
                                  'cat_cols.',
                     _NOT_VARIABLE_KEY: 'Following features are not variable: ',
                     _NOT_NUMERIC_KEY: 'Following features are not numeric: ',
                     _CLASS_DISBALANCE: 'There is class disbalance. ' +
                                        'Probably, you can solve it by data ' +
                                        'augmentation.',
                     _TOO_RARE_KEY: 'Following classes are too rare: ',
                     _TOO_COMMON_KEY: 'Following classes are too common: '}

    def __init__(self, print_hints=True,
                 sample_size=None,
                 category_qoute=0.2,
                 sparse_qoute=0.8,
                 class_disbalance_qoute=0.5,
                 pvalue_boundary=0.05):
        """
        Args:
            sample_size (int): number of objects to be used for category
                and sparsity diagnostic. If None, whole data will be used.
            category_qoute (0 < float < 1): max number of distinct feature values in sample
                to assume this feature categorial
            sparse_qoute (0 < float < 1): zeros portion in sample required to assume this
                feature sparse
            class_disbalance_qoute (0 < float < 1): class portion should be distant from the mean
                to assume this class disbalanced
        """
        self._print_hints = print_hints
        self._sample_size = sample_size
        self._category_qoute = category_qoute
        self._sparse_qoute = sparse_qoute
        self._class_disbalance_qoute = class_disbalance_qoute
        self._pvalue_boundary = pvalue_boundary

    def _preproc_params(self, X, feature_indexes=None, cast_to=None):
        has_names = isinstance(X, np.ndarray) and (X.dtype.names is not None)
        if feature_indexes is None:
            if has_names:
                feature_indexes = list(X.dtype.names)
            else:
                feature_indexes = np.arange(np.shape(X)[1])
        indexes_is_str = isinstance(feature_indexes[0], str)
        if not has_names and indexes_is_str:
            raise ValueError("If feature_indexes is a list of str " +
                             "X should be a np.ndarray and X.dtype should contain fields")

        if isinstance(X, np.ndarray):
            if has_names:
                if indexes_is_str:
                    features = pd.DataFrame(X[feature_indexes]).values.T
                else:
                    features = pd.DataFrame(X).values.T[feature_indexes]
            else:
                features = X.T[feature_indexes]
        else:
            features = [[obj[ind] for obj in X] for ind in feature_indexes]

        if cast_to is not None and isinstance(features, np.ndarray):
            features = features.astype(cast_to)

        return features, feature_indexes

    def check_categorial(self, X):
        """
        Find category features in X

        Args:
            X (array-like with shape (n_objects, n_features)): features from your dataset
        Return:
            dict of shape::

                {
                    'not numeric': list of feature indexes,
                    'not variable': list of feature indexes
                }

        """
        to_find = Guru._CATEGORIAL
        return self._get_categorial_or_sparse(X, to_find)

    def check_sparse(self, X):
        """
        Find sparse features in X

        Args:
            X (array-like with shape (n_objects, n_features)): features from your dataset
        Return:
            list of features which are supposed to be sparse
        """
        to_find = Guru._SPARSE
        return self._get_categorial_or_sparse(X, to_find)

    def _get_categorial_or_sparse(self, X, to_find):
        if to_find == Guru._CATEGORIAL:
            candidates = defaultdict(list)
        elif to_find == Guru._SPARSE:
            candidates = []
        else:
            raise ValueError('In _get_categorial_or_sparse to_find must be ' +
                             Guru._CATEGORIAL + ' or ' + Guru._SPARSE)

        X, indexes = self._preproc_params(X)
        for feature, i in zip(X, indexes):
            if not (isinstance(feature[0], float)
                    or isinstance(feature[0], int)):
                if to_find == Guru._CATEGORIAL:
                    candidates[Guru._NOT_NUMERIC_KEY].append(i)
            else:
                if (self._sample_size is not None and
                        self._sample_size < len(feature)):
                    sample = np.random.choice(feature,
                                              self._sample_size,
                                              False)
                else:
                    sample = feature
                counter = Counter(sample)

                if to_find == Guru._CATEGORIAL:
                    # remove zeros from sample in order to avoid detecting sparse
                    # features as categorial
                    cat_quote = (len(sample) - counter[0]) * self._category_qoute
                    if len(counter) > 1 and len(counter) - 1 < cat_quote:
                        candidates[Guru._NOT_VARIABLE_KEY].append(i)
                elif counter[0] > len(sample) * self._sparse_qoute:
                    candidates.append(i)

        self._print_warning(candidates, Guru._MESSAGE_DICT[to_find])
        return candidates

    def check_class_disbalance(self, y):
        """
        Find disbalanced classes in y.
        You should use this method only if you are solving classification task

        Args:
            y (array-like with shape (n_objects,)): target classes in your dataset
        Return:
            dict of shape::

                {
                    'too common': list of classes,
                    'too rare': list of classes
                }

        """
        candidates = defaultdict(list)
        counter = Counter(y)
        upper = len(y) / len(counter) / self._class_disbalance_qoute
        lower = len(y) / len(counter) * self._class_disbalance_qoute

        for label, cnt in counter.items():
            if cnt > upper:
                candidates[Guru._TOO_COMMON_KEY].append(label)
            if cnt < lower:
                candidates[Guru._TOO_RARE_KEY].append(label)

        self._print_warning(candidates, Guru._MESSAGE_DICT[Guru._CLASS_DISBALANCE])

        return candidates

    def draw_correlation_heatmap(self, X, feature_indexes=None, figsize=(15, 10), **heatmap_kwargs):
        """
        Draw correlation heatmap between features with specified indexes from X

        Args:
            X (array-like with shape (n_objects x n_features)): features from your dataset
            feature_indexes (list of int or str): features which should be checked for correlation.
                If None all features will be checked.
                If it is list of str X should be a np.ndarray and X.dtype should contain fields
            figsize (tuple of int): Size of figure with heatmap

        """
        heatmap_kwargs.setdefault('cmap', Guru._DEFAULT_CMAP)
        features, feature_indexe = self._preproc_params(X, feature_indexes, cast_to=np.float)

        plt.figure(figsize=figsize)
        seaborn.heatmap(np.corrcoef(features),
                        annot=True, ax=plt.axes(),
                        xticklabels=feature_indexes,
                        yticklabels=feature_indexes,
                        **heatmap_kwargs)
        plt.show()

    def draw_2dhist(self, X, feature_indexes=None, figsize=(6, 4), **hist_kwargs):
        """
        Draw 2dhist for each pair of features with specified indexes

        Args:
            X (array-like with shape (n_objects x n_features)): features from your dataset
            feature_indexes (list of int or str): features which should be checked for correlation.
                If None all features will be checked.
                If it is list of str X should be a np.ndarray and X.dtype should contain fields
            figsize (tuple of int): Size of figure with hist2d

        """
        features, feature_indexes = self._preproc_params(X, feature_indexes, cast_to=np.float)
        hist_kwargs.setdefault('cmap', Guru._DEFAULT_CMAP)
        hist_kwargs.setdefault('bins', len(X) ** 0.5)
        for i, (first_ind, first_feature) in enumerate(zip(feature_indexes[:-1], features[:-1])):
            for second_ind, second_feature in zip(feature_indexes[i + 1:],
                                                  features[i + 1:]):
                plt.figure(figsize=figsize)
                plt.hist2d(first_feature, second_feature, **hist_kwargs)
                plt.title(str((first_ind, second_ind)))
                plt.xlabel(str(first_ind))
                plt.ylabel(str(second_ind))
                plt.show()

    def check_correlation(self, X, feature_indexes=None):
        """
        Find correlated features among features with specified indexes from X

        Args:
            X (array-like with shape (n_objects x n_features)): features from your dataset
            feature_indexes: list of features which should be checked for correlation.
                If None all features will be checked
        Return:
            list of pairs of features which are supposed to be correlated
        """
        features, feature_indexes = self._preproc_params(X, feature_indexes, cast_to=np.float)

        candidates = []
        for i, (first_ind, first_feature) in enumerate(zip(feature_indexes[:-1], features[:-1])):
            for second_ind, second_feature in zip(feature_indexes[i + 1:],
                                                  features[i + 1:]):
                pvalue = ss.spearmanr(first_feature, second_feature)[1]
                if pvalue < self._pvalue_boundary:
                    candidates.append((first_ind, second_ind))

        self._print_warning(candidates, Guru._MESSAGE_DICT[Guru._CORRELATION])
        return candidates

    def _print_warning(self, elements, warning):
        if isinstance(elements, dict):
            for element in elements.values():
                if len(element) > 0:
                    self.no_warnings = False
                    if self._print_hints:
                        print(warning)
                    break
            for key in elements.keys():
                self._print_warning(elements[key], Guru._MESSAGE_DICT[key])
        else:
            if len(elements) > 0:
                self.no_warnings = False
                if self._print_hints:
                    print(warning, elements)

    def check_everything(self, data):
        """
        Full data check. Find category features, sparse features, correlated features and disbalanced classes.

        Args:
            data (XYCDataset-like): your dataset
        Return:
            (categorials, sparse, disbalanced, correlated)

            * categorials: indexes of features which are supposed to be categorial
            * sparse: indexes of features which are supposed to be sparse
            * disbalanced: disbalanced classes
            * correlated: indexes of features which are supposed to be correlated

        For more detailes see methods:

            * check_categorials
            * check_sparse
            * check_class_disbalance
            * check_correlation

        """
        self.no_warnings = True

        sparse = self.check_sparse(data.X)
        categorials = self.check_categorials(data.X)
        disbalanced = self.check_class_disbalance(data.y)
        correlated = self.check_correlation(data.X)

        if self.no_warnings and self._print_hints:
            print('Everything is allright!')

        return categorials, sparse, disbalanced, correlated
