import pytest

from collections import OrderedDict
from skopt.space import Integer
from sklearn.datasets import make_classification, load_breast_cancer
from pandas import DataFrame

from modelgym.metrics import RocAuc, Accuracy
from modelgym.models import XGBClassifier, RFClassifier, LGBMClassifier, CtBClassifier, EnsembleClassifier
from modelgym.trainers import SkoptTrainer
from modelgym.utils import XYCDataset, ModelSpace


# TODO: add LGBMClassifier, EnsembleClassifier
CLASSIFIERS = [CtBClassifier, XGBClassifier, RFClassifier]
CLASSIFIERS_NAMES = dict(zip(
    CLASSIFIERS,
    ['CtBClassifier', 'XGBClassifier', 'RFClassifier'],
))


class Case(object):

    def __init__(self, dataset, expected_accuracy):
        self.dataset = dataset
        self.expected_accuracy = expected_accuracy

basic_X, basic_y = make_classification(
    n_samples=20, n_features=4, n_informative=2, n_classes=2)
basic_XYCDataset = XYCDataset(basic_X, basic_y)
basic_DataFrame = DataFrame(data=basic_X)
basic_DataFrame['y'] = basic_y
path = 'basic_dataset.csv'


@pytest.fixture(scope='session')
def basic_dataset_path(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp('data').join(path))
    basic_DataFrame.to_csv(filename)
    return filename


def models(classifier):
    return ModelSpace(
        classifier,
        space=[Integer(low=10, high=11, name='iterations')],
        space_update=False,
    )


dataset = load_breast_cancer()
test_X, test_y = dataset.data[:, :2], dataset.target
test_XYCDataset = XYCDataset(test_X, test_y)


TEST_CASES = OrderedDict([
    (
        "basic_case_XYCDataset",
        Case(
            dataset=basic_XYCDataset,
            expected_accuracy=0.8,
        )
    ),
    (
        "basic_case_DataFrame",
        Case(
            dataset=basic_DataFrame,
            expected_accuracy=0.8,
        )
    ),
    (
        "test_XYCDataset",
        Case(
            dataset=test_XYCDataset,
            expected_accuracy=0.8,
        )
    )
])


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_basic_pipeline_biclass(classifier):
    test_case = TEST_CASES['basic_case_XYCDataset']
    trainer = SkoptTrainer(models(classifier))
    trainer.crossval_optimize_params(
        Accuracy(), test_case.dataset, metrics=[Accuracy(), RocAuc()])
    trainer.get_best_results()
    results = trainer.get_best_results()[CLASSIFIERS_NAMES[classifier]][
        'result']['metric_cv_results']
    accuracy = sum([metric['accuracy'] for metric in results]) / len(results)
    assert pytest.approx(accuracy, abs=0.2) == test_case.expected_accuracy


@pytest.mark.parametrize(
    'test_case',
    TEST_CASES.values(),
    ids=list(TEST_CASES.keys())
)
def test_basic_pipeline_data_type(test_case):
    classifier = CLASSIFIERS[0]
    trainer = SkoptTrainer(models(classifier))
    trainer.crossval_optimize_params(
        Accuracy(), test_case.dataset, metrics=[Accuracy(), RocAuc()])
    trainer.get_best_results()
    results = trainer.get_best_results()[CLASSIFIERS_NAMES[classifier]][
        'result']['metric_cv_results']
    accuracy = sum([metric['accuracy'] for metric in results]) / len(results)
    assert pytest.approx(accuracy, abs=0.2) == test_case.expected_accuracy


def test_basic_pipeline_path_to_csv(basic_dataset_path):
    classifier = CLASSIFIERS[0]
    trainer = SkoptTrainer(models(classifier))
    trainer.crossval_optimize_params(
        Accuracy(), basic_dataset_path, metrics=[Accuracy(), RocAuc()])
    trainer.get_best_results()
