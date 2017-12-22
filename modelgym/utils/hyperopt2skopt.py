from hyperopt.pyll.stochastic import recursive_set_rng_kwarg
from hyperopt.pyll import Apply

import numpy as np
from skopt.space import Integer, Real, Categorical
from numbers import Number


def is_string(obj):
    return isinstance(obj, str)


def is_number(obj):
    return isinstance(obj, Number)


def is_int(obj):
    return isinstance(obj, int)


class NodeParser(object):
    def parse(self, node):
        raise NotImplemented()

    def get_skopt_dimention(self):
        raise NotImplemented()


class LiteralNodeParser(NodeParser):
    def __init__(self, obj_checker=lambda obj: True,
                 obj_parser=lambda obj: obj):
        self._obj_checker = obj_checker
        self._obj_parser = obj_parser
        self.obj = None

    def parse(self, node):
        if node.name != 'literal':
            return False
        if node.named_args or node.pos_args:
            return False
        if not self._obj_checker(node.obj):
            return False
        self.obj = self._obj_parser(node.obj)
        return True


class DistributionParser(NodeParser):
    def __init__(self, name, param_names, param_checkers):
        self.name = name
        self.param_names = param_names
        self.param_checkers = param_checkers
        self.params = {'distr_name': self.name}

    def _parse_param(self, param_node, param_name, param_checker):
        param_parser = LiteralNodeParser(obj_checker=is_number)
        if not param_parser.parse(param_node):
            return False
        self.params[param_name] = param_parser.obj
        return True

    def parse(self, node):
        if node.name != self.name:
            return False
        named_args = {arg[0]: arg[1] for arg in node.named_args}

        for arg_num, arg_name in enumerate(self.param_names):
            arg_checker = self.param_checkers[arg_num]
            if len(node.pos_args) >= arg_num + 1:
                if arg_name in node.named_args:
                    return False
                param_node = node.pos_args[arg_num]
            else:

                if not arg_name in named_args:
                    return False
                param_node = named_args[arg_name]

            if not self._parse_param(param_node, arg_name, arg_checker):
                return False
        return True


class UniformParser(DistributionParser):
    def __init__(self):
        super().__init__('uniform', ['low', 'high'], [is_number, is_number])


class RandintParser(DistributionParser):
    def __init__(self):
        super().__init__('randint', ['upper'], [is_int])


def node2distribution_parser(distribution_node):
    name = distribution_node.name
    if name == 'uniform':
        return UniformParser()
    if name == 'randint':
        return RandintParser()
    return None


class HyperoptParamParser(NodeParser):
    def __init__(self):
        self.name = None
        self.params = None

    def parse(self, node):
        if node.name != 'hyperopt_param' or node.named_args or \
           len(node.pos_args) != 2:
            return False

        name_node = node.pos_args[0]
        name_parser = LiteralNodeParser(obj_checker=is_string)
        if not name_parser.parse(name_node):
            return False
        self.name = name_parser.obj

        distribution_node = node.pos_args[1]
        distribution_parser = node2distribution_parser(distribution_node)
        if distribution_parser is None:
            return False
        if not distribution_parser.parse(distribution_node):
            return False
        self.params = distribution_parser.params

        return True

    def get_skopt_dimention(self):
        if self.params['distr_name'] == 'randint':
            return Integer(0, self.params['upper'])
        return None


class FloatParser(NodeParser):
    def __init__(self):
        self.name = None
        self.params = None

    def parse(self, node):
        if node.name != 'float' or node.named_args or len(node.pos_args) != 1:
            return False
        hyperopt_param_node = node.pos_args[0]
        hyperopt_param_parser = HyperoptParamParser()
        if not hyperopt_param_parser.parse(hyperopt_param_node):
            return False
        self.name = hyperopt_param_parser.name
        self.params = hyperopt_param_parser.params
        return True

    def get_skopt_dimention(self):
        return Real(self.params['low'], self.params['high'])


class IntParser(NodeParser):
    def __init__(self):
        self.name = None
        self.params = None

    def parse(self, node):
        if node.name != 'int' or node.named_args or len(node.pos_args) != 1:
            return False

        # sometimes hyperopt wraps it's distribution into 'float',
        # that's why node tree looks like int->float->distribution

        float_parser = FloatParser()
        if float_parser.parse(node.pos_args[0]):
            self.name = float_parser.name
            self.params = float_parser.params
            return True

        hyperopt_param_parser = HyperoptParamParser()
        if not hyperopt_param_parser.parse(node.pos_args[0]):
            return False
        self.name = hyperopt_param_parser.name
        self.params = hyperopt_param_parser.params
        return True

    def get_skopt_dimention(self):
        if self.params['distr_name'] == 'uniform':
            low = int(self.params['low'])
            high = int(self.params['high'])
            return Integer(low, high)
        if self.params['distr_name'] == 'randint':
            return Integer(0, self.params['upper'])


class SwitchParser(NodeParser):
    def __init__(self):
        self.name = None
        self.options = None

    def parse(self, node):
        if node.name != 'switch' or len(node.pos_args) < 2 or \
           node.named_args:
            return False
        hyperopt_param_parser = HyperoptParamParser()
        if not hyperopt_param_parser.parse(node.pos_args[0]):
            return False
        params = hyperopt_param_parser.params
        if params['distr_name'] != 'randint' or \
           params['upper'] != len(node.pos_args) - 1:
           return False
        self.name = hyperopt_param_parser.name
        options = []
        for option_node in node.pos_args[1:]:
            literal_parser = LiteralNodeParser()
            if not literal_parser.parse(option_node):
                return False
            options.append(literal_parser.obj)
        self.options = options
        return True

    def get_skopt_dimention(self):
        return Categorical(self.options, transform="identity")


def node2sampled_dimention(node, rng, sample_size):
    recursive_set_rng_kwarg(node, rng)
    samples = [node.eval() for _ in range(sample_size)]
    return Categorical(samples, transform="identity")


def node2supported_dimention(node):
    for name, parser_class in [('int', IntParser),
                               ('float', FloatParser),
                               ('switch', SwitchParser),
                               ('hyperopt_param', HyperoptParamParser)]:
        if node.name != name:
            continue

        parser = parser_class()
        if parser.parse(node):
            dimention = parser.get_skopt_dimention()
            return dimention
    return None


def hyperopt2skopt_space(hyperopt_space, sample_size=1000):
    skopt_space = []
    ind2names = []
    rng = np.random.RandomState()
    for node_name, node in hyperopt_space.items():
        ind2names.append(node_name)
        if not isinstance(node, Apply):
            skopt_space.append(Categorical([node], transform='identity'))
            continue
        dimention = node2supported_dimention(node)
        if dimention is None:
            dimention = node2sampled_dimention(node, rng, sample_size)
        skopt_space.append(dimention)
    return skopt_space, ind2names
