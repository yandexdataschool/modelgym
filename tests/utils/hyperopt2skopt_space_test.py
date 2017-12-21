import pytest

from modelgym.utils import hyperopt2skopt_space

import numpy as np
from skopt.space import Integer, Real, Categorical
from hyperopt import hp
from hyperopt.pyll.base import scope


def test_hyperopt2skopt_space():
    hyperopt_space = {
        'int_uniform': scope.int(hp.uniform('l_int_uniform', 1, 7)),
        'randint': hp.randint('l_randint', 7),
        'uniform': hp.uniform('l_uniform', -3, 3),
        'uniform_named': hp.uniform('l_uniform_named', low=1, high=10),
        'uniform_part_named': hp.uniform('l_uniform_part_named', 1, high=10),
        'unsupported': hp.loguniform('l_unsupported', -1, 5),
        'choice': hp.choice('choice', ['a', 'b', 4]),
        'random_param': 'just_one_val',
    }

    space, ind2names = hyperopt2skopt_space(hyperopt_space, sample_size=100)
    assert len(space) == len(ind2names)
    named_space = {ind2names[i]: space[i] for i in range(len(space))}

    int_uniform = named_space['int_uniform']
    assert isinstance(int_uniform, Integer)
    assert int_uniform.low == 1
    assert int_uniform.high == 7

    randint = named_space['randint']
    assert isinstance(randint, Integer)
    assert randint.low == 0
    assert randint.high == 7

    uniform = named_space['uniform']
    assert isinstance(uniform, Real)
    assert uniform.low == -3
    assert uniform.high == 3

    uniform_named = named_space['uniform_named']
    assert isinstance(uniform_named, Real)
    assert uniform_named.low == 1
    assert uniform_named.high == 10

    uniform_part_named = named_space['uniform_part_named']
    assert isinstance(uniform_part_named, Real)
    assert uniform_part_named.low == 1
    assert uniform_part_named.high == 10

    unsupported = named_space['unsupported']
    assert isinstance(unsupported, Categorical)
    assert len(unsupported.categories) == 100
    assert all([np.exp(-1) <= x <= np.exp(5) for x in unsupported.categories])

    choice = named_space['choice']
    assert isinstance(choice, Categorical)
    assert set(choice.categories) == {'a', 'b', 4}

    random_param = named_space['random_param']
    assert isinstance(random_param, Categorical)
    assert set(random_param.categories) == {'just_one_val'}
