from collections import Counter, defaultdict

import numpy as np


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

    _MESSAGE_DICT = {_SPARSE: 'Consider use hashing trick for ' +
                              'your sparse features, if you haven\'t ' +
                              'already. Following features are ' +
                              'supposed to be sparse: ',
                     _CATEGORIAL: 'Some features are ' +
                                  'supposed to be categorial. Make sure ' +
                                  'that all categorial features are in ' +
                                  'cat_cols',
                     _NOT_VARIABLE_KEY: 'Following features are not variable: ',
                     _NOT_NUMERIC_KEY: 'Following features are not numeric: ',
                     _CLASS_DISBALANCE: 'There is class disbalance. ' +
                                        'Probably, you can solve it by data ' +
                                        'augmentation',
                     _TOO_RARE_KEY: 'Following classes are too rare: ',
                     _TOO_COMMON_KEY: 'Following classes are too common: '}

    def __init__(self, print_hints=True,
                 sample_size=None,
                 category_qoute=0.2,
                 sparse_qoute=0.8,
                 class_disbalance_qoute=0.5):
        """
        Arguments:
            sample_size: int
                number of objects to be used for category
                and sparsity diagnostic. If None, whole data will be used.
            category_qoute: 0 < float < 1
                max number of distinct feature values in sample
                to assume this feature categorial
            sparse_qoute: 0 < float < 1
                zeros portion in sample required to assume this
                feature sparse
            class_disbalance_qoute: 0 < float < 1
                class portion should be distant from the mean
                to assume this class disbalanced
        """
        self._print_hints = print_hints
        self._sample_size = sample_size
        self._category_qoute = category_qoute
        self._sparse_qoute = sparse_qoute
        self._class_disbalance_qoute = class_disbalance_qoute

    def check_categorial(self, X):
        """
        Arguments:
            X: array-like with shape (n_objects x n_features)
        Returns:
            out: dict
                out['""" + Guru._NOT_NUMERIC_KEY + """']: list
                    indexes of features which aren't numeric
                out['""" + Guru._NOT_VARIABLE_KEY + """']: list
                    indexes of features which are supposed to be not variable
        """
        to_find = Guru._CATEGORIAL
        return self._get_categorial_or_sparse(X, to_find)

    def check_sparse(self, X):
        """
        Arguments:
            X: array-like with shape (n_objects x n_features)
        Returns:
            out: list
                features which are supposed to be sparse
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

        for i in range(len(X[0])):
            feature = [obj[i] for obj in X]
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
        Arguments:
            y: array-like with shape (n_objects,)
        Returns:
            out['""" + Guru._TOO_COMMON_KEY + """']: list
                too common classes
            out['""" + Guru._TOO_RARE_KEY + """']: list
                too rare classes
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
        Arguments:
            data: XYCDataset-like
        Returns:
            (categorials, sparse, disbalanced)
                categorials: indexes of features which are supposed to be categorial
                sparse: indexes of features which are supposed to be sparse
                disbalanced: disbalanced classes
            For more detailes see respective methods:
                check_categorials
                check_sparse
                check_class_disbalance
        """

        categorials, sparse, disbalanced = [None] * 3
        self.no_warnings = True
        if self._check_sparsity:
            sparse = self.check_sparse(data.X)

        if self._check_categorial_features:
            categorials = self.check_categorials(data.X)

        if self._check_class_disbalance:
            disbalanced = self.check_class_disbalance(data.y)

        if self.no_warnings and self._print_hints:
            print('Everything is allright!')

        return categorials, sparse, disbalanced
