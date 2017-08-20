import pytest
import sklearn
import xgboost as xgb
from sklearn.cross_validation import train_test_split

from modelgym.util import split_and_preprocess
import modelgym


@pytest.fixture
def read_data():
    iris = sklearn.datasets.load_iris()
    return iris  # data and target


def test_preprocess_cat_cols():
    return 0


def test_elementwise_loss():
    return 0


def test_split_and_preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.5)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=2)

    return 0
