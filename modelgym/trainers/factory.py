from modelgym.trainers.hyperopt_trainers import TPETrainer, RandomTrainer
from modelgym.trainers.skopt_trainers import GPTrainer, ForestTrainer

SUPPORTED_TRAINERS = {
    'tpe' : TPETrainer,
    'random' : RandomTrainer,
    'gp' : GPTrainer,
    'forest' : ForestTrainer
}

def get_trainer_by_algo(algo_name, *args, **kwargs):
    if algo_name not in SUPPORTED_TRAINERS:
        raise ValueError("algo name should be one of %s" % str(list(SUPPORTED_TRAINERS.keys())))
    return SUPPORTED_TRAINERS[algo_name](*args, **kwargs)
