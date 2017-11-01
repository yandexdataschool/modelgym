from collections import Counter, defaultdict

import numpy as np


class Guru:
    def __init__(self, category_counter_size=100,
                 category_qoute=0.2,
                 sparse_qoute=0.2,
                 class_balance_restriction=0.5,
                 check_categorial_features=True,
                 check_class_disbalance=True,
                 check_sparsity=True):
        self._category_counter_size = category_counter_size
        self._category_qoute = category_qoute
        self._sparse_qoute = sparse_qoute
        self._class_balance_restriction = class_balance_restriction
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
                if self._category_counter_size < len(feature):
                    sample = np.random.choice(feature,
                                              self._category_counter_size,
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
        upper = len(y) / len(counter) / self._class_balance_restriction
        lower = len(y) / len(counter) * self._class_balance_restriction

        for label, cnt in counter.items():
            if cnt > upper:
                candidates['too common'].append(label)
            if cnt < lower:
                candidates['too rare'].append(label)

        return candidates

    def _get_sparce_features(self):
        pass

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
        if self._check_categorial_features or self._check_sparsity:
            categorials, sparse = self._get_categorial_and_sparce(data.X)
            if self._check_sparsity:
                self._print_warning(sparse, 'Consider use hashing trick for ' +
                                    'your sparse features, if you haven\'t ' +
                                    'already. Following features are ' +
                                    'supposed to be spares: ')
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
                                'Following classes are too rare: ')
