import os
import pytest
import shutil

from hyperopt import hp
from sklearn.datasets import make_classification, make_regression

from modelgym.metrics import RocAuc, Accuracy, Mse
from modelgym.models import XGBClassifier, RFClassifier, LGBMClassifier, CtBClassifier, \
                            XGBRegressor, LGBMRegressor, CtBRegressor, LearningTask, \
                            EnsembleClassifier, EnsembleRegressor
from modelgym.trackers import LocalTracker
from modelgym.trainers import TpeTrainer, RandomTrainer, RFTrainer, GPTrainer
from modelgym.utils import XYCDataset, ModelSpace, train_ensemble_model


TRAINER_CLASS = [TpeTrainer, None]  # RandomTrainer uses if None
TRACKABLE_TRAINER_CLASS = [TpeTrainer, None]


@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_biclass(trainer_class):
    x, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)

    results = train_ensemble_model(
        [XGBClassifier, LGBMClassifier, RFClassifier],
        EnsembleClassifier,
        metric=RocAuc(),
        X_train=x,
        y_train=y,
        split_size=0.2,
        random_state=0,
        base_trainer=trainer_class,
        ensemble_trainer=trainer_class
    )
    assert 'final_model' in results
    assert 'base_training' in results
    assert 'ensemble_training' in results
    assert isinstance(results['final_model'], EnsembleClassifier)


@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_regression(trainer_class):
    x, y = make_regression(n_samples=200, n_features=20,
                           n_informative=10, n_targets=1)

    xgb_model_space = ModelSpace(XGBRegressor, {'n_estimators': 15}, name='XGB')
    ctb_model_space = ModelSpace(CtBRegressor, {
                'learning_rate': hp.loguniform('learning_rate', -5, -1),
                'iterations': 10
            })

    results = train_ensemble_model(
        [xgb_model_space, LGBMClassifier, ctb_model_space],
        EnsembleRegressor,
        metric=Mse(),
        dtrain=XYCDataset(x, y),
        split_size=0.2,
        random_state=0,
        base_trainer=trainer_class,
        ensemble_trainer=trainer_class,
        base_trainer_kwargs={'opt_evals': 3},
        ensemble_trainer_kwargs={'opt_evals': 3}
    )
    assert 'final_model' in results
    assert 'base_training' in results
    assert 'ensemble_training' in results
    assert isinstance(results['final_model'], EnsembleRegressor)

@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_biclass(trainer_class):
    x, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)

    results = train_ensemble_model(
        [XGBClassifier, LGBMClassifier, RFClassifier],
        XGBClassifier,
        metric=RocAuc(),
        X_train=x,
        y_train=y,
        split_size=0.2,
        random_state=0,
        base_trainer=trainer_class,
        ensemble_trainer=trainer_class,
        add_meta_features=True,
        base_trainer_kwargs={'opt_evals': 3},
        ensemble_trainer_kwargs={'opt_evals': 3},
    )
    assert 'final_model' in results
    assert 'base_training' in results
    assert 'ensemble_training' in results
    assert isinstance(results['final_model'], XGBClassifier)


@pytest.mark.parametrize("trainer_class", TRACKABLE_TRAINER_CLASS)
def test_advanced_pipeline_biclass(trainer_class):
    try:
        x, y = make_classification(n_samples=200, n_features=20,
                                   n_informative=10, n_classes=2)
        directory = '/tmp/local_dir'

        results = train_ensemble_model(
            [XGBClassifier, LGBMClassifier, RFClassifier],
            EnsembleClassifier,
            metric=RocAuc(),
            X_train=x,
            y_train=y,
            split_size=0.1,
            random_state=0,
            base_trainer=trainer_class,
            ensemble_trainer=trainer_class,
            add_meta_features=False,
            save_dir=directory,
            base_trainer_kwargs={'metrics': [RocAuc()], 'opt_evals': 3},
            ensemble_trainer_kwargs={'metrics': [RocAuc()], 'opt_evals': 3}
        )
        assert 'final_model' in results
        assert 'base_training' in results
        assert 'ensemble_training' in results
        assert isinstance(results['final_model'], EnsembleClassifier)

        assert os.listdir(directory)
    except Exception as e:
        try:
            shutil.rmtree(directory)
        except Exception as _:
            pass
        raise e
