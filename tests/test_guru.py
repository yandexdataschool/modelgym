from modelgym import Guru
from sklearn.datasets import load_breast_cancer
from collections import namedtuple


XYCDataset = namedtuple('data', ['X', 'y', 'cat_cols'])
X = [['joke', 1231, 0.12312, 1, True, 0, 0],
     ['lol', 123, 1231.1231, 1, False, 0, 0],
     ['not joke', 1313, 12.133, 1, False, 1, 0],
     ['joke', 12312, 0.24183, 0, True, 0, 0],
     ['silly joke', 321, 0.12, 1, False, 0, 0],
     ['silly joke', 32, 0.2, 1, False, 0, 0],
     ['silly joke', 31, 0.1, 1, True, 0, 1.1],
     ['silly joke', 21, 0.123, 1, True, 0, 4.6],
     ['silly joke', 21, 0.123, 1, True, 0, 0],
     ['silly joke', 1, 0.124, 1, True, 0, -2.73]]
y = [0]*8 + [1] + [3]*2
_TOY_DATASET = XYCDataset(X, y, None)


def _empty(obj):
    if isinstance(obj, dict):
        for i in obj.values():
            if not _empty(i):
                return False
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in obj:
            if not _empty(i):
                return False
    else:
        return True
    return True


def test_without_warnings():
    guru = Guru(print_hints=False)
    X, y = load_breast_cancer(True)
    result = guru.check_data(XYCDataset(X, y, None))
    assert _empty(result)


def test_warnings():
    guru = Guru(print_hints=False)
    result = guru.check_data(_TOY_DATASET, category_qoute=0.5)
    assert result == ({'not numeric': [0], 'not variable': [3, 4]},
                      [5], {'too rare': [1], 'too common': [0]})


def test_sparse_arguments():
    guru = Guru(print_hints=False)

    params = [{'check_sparsity': False},
              {'sparse_qoute': 0.3}]
    answers = [(None, None, None),
               (None, [5, 6], None)]

    for param, answer in zip(params, answers):
        assert answer == guru.check_data(_TOY_DATASET,
                                         check_categorial_features=False,
                                         check_class_disbalance=False,
                                         **param)
