import os
import pytest
import shutil

from hyperopt import hp
from sklearn.datasets import make_classification, make_regression

from modelgym.metrics import RocAuc, Accuracy, Mse
from modelgym.models import XGBClassifier, RFClassifier, LGBMClassifier, CtBClassifier, \
                            XGBRegressor, LGBMRegressor, CtBRegressor, \
                            EnsembleClassifier, EnsembleRegressor
from modelgym.trackers import LocalTracker
from modelgym.trainers import TpeTrainer, RandomTrainer, RFTrainer, GPTrainer
from modelgym.utils import XYCDataset, ModelSpace


TRAINER_CLASS = [TpeTrainer, RandomTrainer, GPTrainer, RFTrainer]
TRACKABLE_TRAINER_CLASS = [TpeTrainer, RandomTrainer]


@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_biclass(trainer_class):
    x, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)

    ctb_model_space = ModelSpace(CtBClassifier, {
                'learning_rate': hp.loguniform('learning_rate', -5, -1),
                'iterations': 10
            })

    trainer = trainer_class([XGBClassifier, LGBMClassifier, RFClassifier,
                             ctb_model_space, EnsembleClassifier])
    dataset = XYCDataset(x, y)
    trainer.crossval_optimize_params(Accuracy(), dataset, opt_evals=3)
    trainer.get_best_results()


@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_regression(trainer_class):
    x, y = make_regression(n_samples=200, n_features=20,
                           n_informative=10, n_targets=1)
    xgb_model_space = ModelSpace(XGBRegressor, {'n_estimators': 15}, name='XGB')
    ctb_model_space = ModelSpace(CtBRegressor, {
                'learning_rate': hp.loguniform('learning_rate', -5, -1),
                'iterations': 10
            })
    trainer = trainer_class([LGBMRegressor, xgb_model_space,
                             ctb_model_space, EnsembleRegressor])
    dataset = XYCDataset(x, y)
    trainer.crossval_optimize_params(Mse(), dataset, opt_evals=3)
    results = trainer.get_best_results()
    assert results['XGB']['result']['params']['n_estimators'] == 15


@pytest.mark.parametrize("trainer_class", TRAINER_CLASS)
def test_basic_pipeline_biclass_with_cat_preprocess_mask(trainer_class):
    x, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)
    trainer = trainer_class([XGBClassifier, LGBMClassifier, RFClassifier])
    dataset = XYCDataset(x, y)
    trainer.crossval_optimize_params(Accuracy(), dataset, opt_evals=3,
                                     cat_preprocess=[False, False, True])
    trainer.get_best_results()


@pytest.mark.parametrize("trainer_class", TRACKABLE_TRAINER_CLASS)
def test_advanced_pipeline_biclass(trainer_class):
    try:
        x, y = make_classification(n_samples=200, n_features=20,
                                   n_informative=10, n_classes=2)
        directory = '/tmp/local_dir'
        tracker = LocalTracker(directory)

        ctb_model_space = ModelSpace(CtBClassifier, {
                    'learning_rate': hp.loguniform('learning_rate', -5, -1),
                    'iterations': 10
                })

        trainer = trainer_class([XGBClassifier, LGBMClassifier, RFClassifier,
                                 ctb_model_space, EnsembleClassifier],
                                tracker=tracker)
        dataset = XYCDataset(x, y)
        trainer.crossval_optimize_params(Accuracy(), dataset, opt_evals=3,
                                         metrics=[RocAuc()])
        trainer.get_best_results()

        assert os.listdir(directory)
    except Exception as e:
        try:
            shutil.rmtree(directory)
        except Exception as _:
            pass
        raise e
