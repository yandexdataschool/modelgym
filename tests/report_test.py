import pytest

from modelgym.metrics import Accuracy
from modelgym.models.rf_model import RFClassifier
from modelgym.models.xgboost_model import XGBClassifier
from modelgym.report import Report
from modelgym.utils.dataset import XYCDataset
from modelgym.utils.model_space import ModelSpace
from sklearn.datasets import make_classification


@pytest.mark.skip(reason="need to test without calling plt.show() method")
def test_basic_pipeline_biclass():
    results = {
        'XGBClassifier': {
            'result': {
                'loss': 0.2655008831920376,
                'loss_variance': 0.0044628124677921106,
                'metric_cv_results': [{
                    'accuracy': 0.73013493253373318
                }, {
                    'accuracy': 0.74062968515742134
                }, {
                    'accuracy': 0.73273273273273276
                }],
                'status': 'ok',
                'params': {
                    'n_estimators': 20,
                    'max_depth': 7
                }
            },
            'model_space': ModelSpace(XGBClassifier)
        },
        'RFClassifier': {
            'result': {
                'loss': 0.2655008831920376,
                'loss_variance': 0.0044628124677921106,
                'metric_cv_results': [{
                    'accuracy': 0.63013493253373318
                }, {
                    'accuracy': 0.64062968515742134
                }, {
                    'accuracy': 0.63273273273273276
                }],
                'status': 'ok',
                'params': {
                    'n_estimators': 10,
                    'max_depth': 5
                }
            },
            'model_space': ModelSpace(RFClassifier)
        }
    }
    x, y = make_classification()
    reporter = Report(results, XYCDataset(x, y), [Accuracy()])
    reporter.summary()
