import pytest

from modelgym.models import XGBClassifier, RFClassifier, LGBMClassifier, \
                            XGBRegressor, LGBMRegressor
from modelgym.trainers import HyperoptTrainer
from modelgym.metrics import RocAuc, Accuracy, Mse
from modelgym.utils import XYCDataset
from modelgym.trackers import LocalTracker

import os
import shutil

from sklearn.datasets import make_classification, make_regression


def test_basic_pipeline_biclass():
    X, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)
    trainer = HyperoptTrainer([XGBClassifier, LGBMClassifier, RFClassifier])
    dataset = XYCDataset(X, y)
    trainer.crossval_optimize_params(Accuracy(), dataset, opt_evals=3)
    trainer.get_best_results()

def test_basic_pipeline_regression():
    X, y = make_regression(n_samples=200, n_features=20,
                           n_informative=10, n_targets=1)
    trainer = HyperoptTrainer([LGBMRegressor])
    dataset = XYCDataset(X, y)
    trainer.crossval_optimize_params(Mse(), dataset, opt_evals=3)
    trainer.get_best_results()

def test_advanced_pipeline_biclass():
    X, y = make_classification(n_samples=200, n_features=20,
                               n_informative=10, n_classes=2)
    DIR = '/tmp/local_dir'
    tracker = LocalTracker(DIR)
    trainer = HyperoptTrainer([XGBClassifier, LGBMClassifier, RFClassifier],
                              tracker=tracker)
    dataset = XYCDataset(X, y)
    trainer.crossval_optimize_params(Accuracy(), dataset, opt_evals=3,
                                     metrics=[RocAuc()])
    trainer.get_best_results()

    assert os.listdir(DIR)

    shutil.rmtree(DIR)
