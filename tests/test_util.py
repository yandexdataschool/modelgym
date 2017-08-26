import pytest
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

from modelgym.util import split_and_preprocess

TEST_SIZE = 0.2
N_CV_SPLITS = 2
NROWS = 1000
N_PROBES = 2
N_ESTIMATORS = 100
TEST_PARAMS = ["classification", "range", "regression"]
APPROVED_PARAMS = ["classification", "regression"]


@pytest.fixture
def read_data():
    iris = load_iris()
    return iris  # data and target


def test_split_and_preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=2)
    # TODO: more tests
    assert len(X_test) <= 0.2 * len(iris_data.data) <= len(X_train) <= 0.8 * len(iris_data.data)
    assert len(y_test) <= 0.2 * len(iris_data.data) <= len(y_train) <= 0.8 * len(iris_data.data)
