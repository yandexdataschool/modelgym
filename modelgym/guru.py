from collections import Counter, defaultdict

import numpy as np


class Guru:
    """
    This class analyze data trying to find some issues.
    """
    def __init__(self, print_hints=True):
        self._print_hints = print_hints

    def _get_categorial_and_sparse(self, X, sample_size, category_qoute, sparse_qoute):
        categorial_candidates = defaultdict(list)
        sparse_candidates = []
        for i in range(len(X[0])):
            feature = [obj[i] for obj in X]
            if not (isinstance(feature[0], float)
                    or isinstance(feature[0], int)):
                categorial_candidates['not numeric'].append(i)
            else:
                if (sample_size is not None and
                        sample_size < len(feature)):
                    sample = np.random.choice(feature,
                                              sample_size,
                                              False)
                else:
                    sample = feature
                counter = Counter(sample)
                # remove zeros from sample in order to avoid detecting sparse
                # features as categorial
                cat_quote = (len(sample) - counter[0]) * category_qoute
                if len(counter) > 1 and len(counter) - 1 < cat_quote:
                    categorial_candidates['not variable'].append(i)
                elif counter[0] > len(sample) * sparse_qoute:
                    sparse_candidates.append(i)

        return categorial_candidates, sparse_candidates

    def _get_disbalanced_classes(self, y, class_balance_qoute):
        candidates = defaultdict(list)
        counter = Counter(y)
        upper = len(y) / len(counter) / class_balance_qoute
        lower = len(y) / len(counter) * class_balance_qoute

        for label, cnt in counter.items():
            if cnt > upper:
                candidates['too common'].append(label)
            if cnt < lower:
                candidates['too rare'].append(label)

        return candidates

    def _print_warning(self, elements, warning):
        if isinstance(elements, dict):
            for element in elements.values():
                if len(element) > 0:
                    self.no_warnings = False
                    if self._print_hints:
                        print(warning)
                    break
        else:
            if len(elements) > 0:
                self.no_warnings = False
                if self._print_hints:
                    print(warning, elements)

    def check_data(self, data, sample_size=None,
                   category_qoute=0.2,
                   sparse_qoute=0.8,
                   class_balance_qoute=0.5,
                   check_categorial_features=True,
                   check_class_disbalance=True,
                   check_sparsity=True):
        """
        Arguments:
            data: XYCDataset-like
            data.X: array-like with shape (n_objects x n_features)
            daya.y: array-like with shape n_objects
            sample_size: int
                number of objects to be used for category
                and sparsity diagnostic. If None, whole data will be used.
            category_qoute: 0 < float < 1
                max number of distinct feature values in sample
                to assume this feature categorial
            sparse_qoute: 0 < float < 1
                zeros portion in sample required to assume this
                feature sparse
            class_balance_qoute: 0 < float < 1
                class portion should be distant from the mean
                to assume this class disbalanced
            check_categorial_features: bool
                check if data.X has categorial features
            check_sparsity: bool
                check if data.X has sparse features
            check_class_disbalance: bool
                check if data.y has class disbalance

        Returns:
            (categorials, sparse, disbalanced)
                categorials: indexes of features which are supposed to be categorial
                sparse: indexes of features which are supposed to be sparse
                disbalanced: disbalanced classes
        """

        categorials, sparse, disbalanced = [None] * 3
        self.no_warnings = True
        if check_categorial_features or check_sparsity:
            _categorials, _sparse = self._get_categorial_and_sparse(data.X,
                                                                    sample_size,
                                                                    category_qoute,
                                                                    sparse_qoute)
            if check_sparsity:
                sparse = _sparse
                self._print_warning(sparse, 'Consider use hashing trick for ' +
                                    'your sparse features, if you haven\'t ' +
                                    'already. Following features are ' +
                                    'supposed to be sparse: ')
            if check_categorial_features:
                categorials = _categorials
                self._print_warning(categorials, 'Some features are ' +
                                    'supposed to be categorial. Make sure ' +
                                    'that all categorial features are in ' +
                                    'cat_cols')
                self._print_warning(categorials['not numeric'],
                                    'Following features are not numeric: ')
                self._print_warning(categorials['not variable'],
                                    'Following features are not variable: ')

        if check_class_disbalance:
            disbalanced = self._get_disbalanced_classes(data.y, class_balance_qoute)
            self._print_warning(disbalanced, 'There is class disbalance. ' +
                                'Probably, you can solve it by data ' +
                                'augmentation')
            self._print_warning(disbalanced['too common'],
                                'Following classes are too common: ')
            self._print_warning(disbalanced['too rare'],
                                'Following classes are too rare: ')

        if self.no_warnings and self._print_hints:
            print('Everything is allright!')

        return categorials, sparse, disbalanced
