import numpy as np

from modelgym import Guru
from sklearn.datasets import load_breast_cancer


np.random.seed(0)
_TOY_X = [['joke', 1231, 0.12312, 1, True, 0, 0],
          ['lol', 123, 1231.1231, 1, False, 0, 0],
          ['not joke', 1313, 12.133, 1, False, 0, 0],
          ['joke', 12312, 0.24183, 0, True, 1, 0],
          ['silly joke', 321, 0.12, 1, False, 0, 0],
          ['silly joke', 32, 0.2, 1, False, 0, 0], ['silly joke', 31, 0.1, 1, True, 0, 1.1],
          ['silly joke', 21, 0.123, 1, True, 0, 4.6],
          ['silly joke', 21, 0.123, 1, True, 0, 0],
          ['joke', 1, 0.23, 1, False, 0, 0],
          ['silly joke', 1, 0.124, 1, True, 0, -2.73]]

N = 100
_NAMED_X = np.zeros((N,), dtype=[('str', 'U25'),
                                 ('categorial', int),
                                 ('sparse', float),
                                 ('corr_1', float),
                                 ('corr_2', float),
                                 ('independent', float)])
_NAMED_X['str'] = 'joke'
_NAMED_X['categorial'] = np.random.binomial(3, 0.6, size=N)
_NAMED_X['sparse'] = np.random.binomial(1, 0.05, size=N) * np.random.normal(size=N)
_NAMED_X['corr_1'] = np.random.normal(size=N)
_NAMED_X['corr_2'] = _NAMED_X['corr_1'] * 50 - 100
_NAMED_X['independent'] = np.random.normal(size=N)

_TOY_Y = [0]*8 + [1] + [3]*2
_BREAST_X, _BREAST_Y = load_breast_cancer(True)


def test_sparse():
    guru_params = [{}, {}, {'sparse_qoute': 0.5}, {}]
    xs = [[_BREAST_X]] + [[_TOY_X]] * 2 + [[_NAMED_X]]
    answers = [[], [5], [5, 6], ['sparse']]

    _iterate_method_test(Guru.check_sparse,
                         init_dicts=guru_params,
                         method_args=xs,
                         expected_results=answers)


def test_categorial():
    guru_params = [{}, {}, {'category_qoute': 0.15}, {}]
    xs = [[_BREAST_X]] + [[_TOY_X]] * 2 + [[_NAMED_X]]
    answers = [{}, {Guru._NOT_NUMERIC_KEY: [0], Guru._NOT_VARIABLE_KEY: [3, 4]},
               {Guru._NOT_NUMERIC_KEY: [0], Guru._NOT_VARIABLE_KEY: [3]},
               {Guru._NOT_NUMERIC_KEY: ['str'], Guru._NOT_VARIABLE_KEY: ['categorial']}]
    _iterate_method_test(Guru.check_categorial,
                         init_dicts=guru_params,
                         method_args=xs,
                         expected_results=answers)


def test_class_disbalance():
    guru_params = [{}, {}, {'class_disbalance_qoute': 0.4},
                   {'class_disbalance_qoute': 0.8}]
    ys = [[_BREAST_Y]] + [[_TOY_Y]] * 2
    answers = [{},
               {Guru._TOO_COMMON_KEY: [0], Guru._TOO_RARE_KEY: [1]},
               {Guru._TOO_RARE_KEY: [1]},
               {Guru._TOO_COMMON_KEY: [0], Guru._TOO_RARE_KEY: [1, 3]}]
    _iterate_method_test(Guru.check_class_disbalance,
                         init_dicts=guru_params,
                         method_args=ys,
                         expected_results=answers)


def test_correlation():
    guru_params = [{}] * 3

    corr_x = np.zeros((N, 3))
    corr_x[:, 0] = np.random.normal(size=N)
    corr_x[:, 1] = -corr_x[:, 0] + np.random.normal(0, 1e-10, size=N)
    corr_x[:, 2] = np.random.normal(size=N)

    args_list = [[_BREAST_X, [0, 1, 2]],
                 [_TOY_X, [2, 3, 5]],
                 [corr_x],
                 [_NAMED_X, ['corr_1', 'corr_2', 'independent']]]

    answers = [[(0, 1), (0, 2), (1, 2)], [(3, 5)], [(0, 1)], [('corr_1', 'corr_2')]]
    _iterate_method_test(Guru.check_correlation,
                         init_dicts=guru_params,
                         method_args=args_list,
                         expected_results=answers)


def _iterate_method_test(method, init_dicts, method_args, expected_results):
    for init_kwargs, args, expected in zip(init_dicts, method_args, expected_results):
        guru = Guru(print_hints=False, **init_kwargs)
        result = method(guru, *args)
        assert result == expected
