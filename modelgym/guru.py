from collections import Counter, defaultdict

import numpy as np


class Guru:
    """
    This class analyze data trying to find some issues.
    """
    def __init__(self, sample_size=None,
                 category_qoute=0.2,
                 sparse_qoute=0.8,
                 class_balance_qoute=0.5,
                 check_categorial_features=True,
                 check_class_disbalance=True,
                 check_sparsity=True):
        """
        Arguments:
            sample_size: how many objects will be used for category
                and sparsity diagnostic. If None whole data will be used.
            category_qoute: upper bound of variants of feature values in sample
                to note this feature as categorial
            sparse_qoute: zeros percentage in sample required to note
                feature as sparse
            class_balance_qoute: how much class percentage should be deflected
                from the mean to note this class as disbalanced
            check_categorial_features: check_data will check if data.X has
                categorial features
            check_sparsity: check_data will check if data.X has sparse features
            check_class_disbalance: check_data will check if data.y has
                class disbalance
        """
        self._sample_size = sample_size
        self._category_qoute = category_qoute
        self._sparse_qoute = sparse_qoute
        self._class_balance_qoute = class_balance_qoute
        self._check_class_disbalance = check_class_disbalance
        self._check_categorial_features = check_categorial_features
        self._check_sparsity = check_sparsity

    def _get_categorial_and_sparce(self, X):
        categorial_candidates = defaultdict(list)
        sparse_candidates = []
        for i in range(len(X[0])):
            feature = [obj[i] for obj in X]
            if not (isinstance(feature[0], float)
                    or isinstance(feature[0], int)):
                categorial_candidates['not numeric'].append(i)
            else:
                if (self._sample_size is not None and
                        self._sample_size < len(feature)):
                    sample = np.random.choice(feature,
                                              self._sample_size,
                                              False)
                else:
                    sample = feature
                counter = Counter(sample)
                # remove zeros from sample in order to avoid detecting sparse
                # features as categorial
                cat_quote = (len(sample) - counter[0]) * self._category_qoute
                if len(sample) != counter[0] and len(counter) - 1 < cat_quote:
                    categorial_candidates['not variable'].append(i)
                elif counter[0] > len(sample) * self._sparse_qoute:
                    sparse_candidates.append(i)

        return categorial_candidates, sparse_candidates

    def _get_disbalanced_classes(self, y):
        candidates = defaultdict(list)
        counter = Counter(y)
        upper = len(y) / len(counter) / self._class_balance_qoute
        lower = len(y) / len(counter) * self._class_balance_qoute

        for label, cnt in counter.items():
            if cnt > upper:
                candidates['too common'].append(label)
            if cnt < lower:
                candidates['too rare'].append(label)

        return candidates

    @staticmethod
    def _print_warning(elements, warning):
        if isinstance(elements, dict):
            for element in elements.values():
                if len(element) > 0:
                    print(warning)
                    break
        else:
            if len(elements) > 0:
                print(warning, elements)

    def check_data(self, data):
        """
        data should be XYCDataset-like
        data.X should has shape (n_objects x n_features)
        daya.y should has shape n_objects
        """
        if self._check_categorial_features or self._check_sparsity:
            categorials, sparse = self._get_categorial_and_sparce(data.X)
            if self._check_sparsity:
                self._print_warning(sparse, 'Consider use hashing trick for ' +
                                    'your sparse features, if you haven\'t ' +
                                    'already. Following features are ' +
                                    'supposed to be sparse: ')
            if self._check_categorial_features:
                self._print_warning(categorials, 'Some features are ' +
                                    'supposed to be categorial. Make sure ' +
                                    'that all categorial features are in ' +
                                    'cat_cols')
                self._print_warning(categorials['not numeric'],
                                    'Following features are not numeric: ')
                self._print_warning(categorials['not variable'],
                                    'Following features are not variable: ')

        if self._check_class_disbalance:
            disbalanced = self._get_disbalanced_classes(data.y)
            self._print_warning(disbalanced, 'There is class disbalance. ' +
                                'Probably, you can solve it by data ' +
                                'augmentation')
            self._print_warning(disbalanced['too common'],
                                'Following classes are too common: ')
            self._print_warning(disbalanced['too rare'],
                                'Followifeaturesng classes are too rare: ')
