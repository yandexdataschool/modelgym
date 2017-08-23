import pytest
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

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


def test_preprocess_cat_cols():
    # not implemented in util.py
    return 0


# TODO: what is it?
def test_elementwise_loss():
    # y=0
    # p=0
    # learning_task=0
    # if learning_task == TASK_CLASSIFICATION:
    #    p_ = np.clip(p, 1e-16, 1-1e-16)
    #    return - y * np.log(p_) - (1 - y) * np.log(1 - p_)
    # return (y - p) ** 2
    return 0


def test_split_and_preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=2)
    #TODO: more samples
    assert len(X_test) <= 0.2 * len(iris_data.data) <= len(X_train) <= 0.8 * len(iris_data.data)
    assert len(y_test) <= 0.2 * len(iris_data.data) <= len(y_train) <= 0.8 * len(iris_data.data)
