import numpy as np
from itertools import compress
from hyperopt import hp
from sklearn.model_selection import train_test_split

from modelgym.trainers import RandomTrainer
from modelgym.trackers import LocalTracker
from modelgym.utils import XYCDataset
from modelgym.utils.model_space import ModelSpace, process_model_spaces


def parse_data_args(dtrain, X_train, y_train, dtrain_2, X_train_2, y_train_2, split_size=0.1, random_state=0):
    if dtrain is None:
        if X_train is None or y_train is None:
            raise RuntimeError('no train data given')
        cat_cols = []
    else:
        if X_train is not None:
            raise RuntimeError('both dtrain and X_train given')
        if y_train is not None:
            raise RuntimeError('both dtrain and y_train given')

        X_train, y_train, cat_cols = dtrain.X, dtrain.y, dtrain.cat_cols

    if dtrain_2 is None and (X_train_2 is None or y_train_2 is None):
        X_train, X_train_2, y_train, y_train_2 = train_test_split(
            X_train, y_train,
            test_size=split_size,
            random_state=random_state
        )
    dtrain = XYCDataset(X_train, y_train, cat_cols)

    if dtrain_2 is None:
        dtrain_2 = XYCDataset(X_train_2, y_train_2, cat_cols)
    else:
        if X_train_2 is not None:
            raise RuntimeError('both dtrain_2 and X_train_2 given')
        if y_train_2 is not None:
            raise RuntimeError('both dtrain_2 and y_train_2 given')

    return dtrain, dtrain_2


def train_ensemble_model(
        model_spaces,
        ensemble_model,
        metric,
        dtrain=None,
        X_train=None,
        y_train=None,
        dtrain_2=None,
        X_train_2=None,
        y_train_2=None,
        split_size=0.2,
        random_state=0,
        base_trainer=None,
        base_trainer_kwargs={},
        ensemble_trainer=None,
        ensemble_trainer_kwargs={},
        save_dir=None,
        base_tracker=None,
        ensemble_tracker=None,
        add_meta_features=False,
        ensemble_model_params={}
    ):
    """
    Args:
        model_spaces (list of modelgym.models.Model or modelgym.utils.ModelSpaces): list of model spaces
                (model classes and parameter spaces to look in). If some list item is Model, it is
                converted in ModelSpace with default space and name equal to model class __name__
        ensemble_model: one of modelgym.models.Model
        metric (modelgym.metrics.Metric): metric to optimize
        dtrain (modelgym.utils.XYCDataset or None): dataset
        X_train (np.array(n_samples, n_features)): instead of dtrain
        y_train (np.array(n_samples)): labels
        # if no *_2 given - split train given above
        dtrain_2 (modelgym.utils.XYCDataset or None): dataset to fit second layer
        X_train_2 (np.array(n_samples, n_features)): instead of dtrain_2
        y_train_2 (np.array(n_samples)): labels to train second layer
        split_size (float 0..1): split train if no train_2 given
        random_state (int): random state to split
        base_trainer (one of modelgym.trainers or None): trainer to train base models
            RandomTrainer uses if None
        base_trainer_kwargs (dict): kwargs to pass to base_trainer.crossval_optimize_params
        ensemble_trainer (one of modelgym.trainers or None): trainer to train second layer model
            RandomTrainer uses if None
        ensemble_trainer_kwargs (dict): kwargs to pass to ensemble_trainer.crossval_optimize_params
        save_dir (str or None): directory to track
        base_tracker (modelgym.trackers or None): tracker for base models
        ensemble_tracker (modelgym.trackers or None): tracker for second layer model
        add_meta_features (bool, default=False): add predictions of base models to train_2
            e.x. make stacking (otherwise - weighted ensemble)
        ensemble_model_params (dict): params for last layer model
    Return:
        dict: {
        'base_training': base_results,  # results of base models trainer
        'ensemble_training': ensemble_results,  # results of ensemble model trainer
        # model instance with all params
        'final_model': ensemble_model(ensemble_results[ensemble_model.__name__]['result']['params'])
    }
    """
    dtrain, dtrain_2 = parse_data_args(dtrain, X_train, y_train,
                                       dtrain_2, X_train_2, y_train_2,
                                       split_size=split_size,
                                       random_state=random_state)

    if save_dir is not None:
        if base_tracker is None:
            base_tracker = LocalTracker(save_dir, 'base_models')
        if ensemble_tracker is None:
            ensemble_tracker = LocalTracker(save_dir, 'ensemble_model')

    if base_trainer is None:
        base_trainer = RandomTrainer(model_spaces, tracker=base_tracker)
    else:
        base_trainer = base_trainer(model_spaces, tracker=base_tracker)

    base_trainer.crossval_optimize_params(metric, dtrain, **base_trainer_kwargs)
    base_results = base_trainer.get_best_results()

    trained_models = []
    for model, space in base_trainer.model_spaces.items():
        trained_models.append(
            space.model_class(base_results[model]['result']['params'])
        )

    if add_meta_features:
        X_train_2, y_train_2, cat_cols = dtrain_2.X, dtrain_2.y, dtrain_2.cat_cols
        meta_features = np.zeros((len(X_train_2), len(model_spaces)))
        if not isinstance(X_train_2, np.ndarray):
            X_train_2 = np.array(X_train_2)
        for i, model in enumerate(trained_models):
            model.fit(dtrain)
            meta_features[:, i] = model.predict(dtrain_2)
        X_train_2 = np.concatenate([X_train_2, meta_features], axis=1)
        dtrain_2 = XYCDataset(X_train_2, y_train_2, cat_cols)

        params = ensemble_model_params
    else:
        params = {
            'weight_{}'.format(i): hp.uniform('weight_{}'.format(i), 0, 1)
            for i, _ in enumerate(model_spaces)
        }
        params['models'] = trained_models

    ensemble_model_space = ModelSpace(ensemble_model, params)

    if ensemble_trainer is None:
        ensemble_trainer = RandomTrainer(ensemble_model_space, tracker=ensemble_tracker)
    else:
        ensemble_trainer = ensemble_trainer(ensemble_model_space, tracker=ensemble_tracker)

    ensemble_trainer.crossval_optimize_params(metric, dtrain_2, **ensemble_trainer_kwargs)
    ensemble_results = ensemble_trainer.get_best_results()

    return {
        'base_training': base_results,
        'ensemble_training': ensemble_results,
        'final_model': ensemble_model(ensemble_results[ensemble_model.__name__]['result']['params'])
    }
