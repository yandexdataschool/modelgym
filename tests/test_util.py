import pytest
from sklearn.model_selection import train_test_split
from modelgym.model import TASK_CLASSIFICATION, TASK_REGRESSION

from modelgym.util import split_and_preprocess

TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
N_CV_SPLITS = 2
APPROVED_PARAMS = [TASK_CLASSIFICATION, TASK_REGRESSION]
TEST_PARAMS = (APPROVED_PARAMS[:])
TEST_PARAMS.append("range")


@pytest.mark.usefixtures("read_data")
def test_split_and_preprocess(read_data):
    iris_data = read_data
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=TEST_SIZE)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train.copy(), y_train,
                                                     X_test.copy(), y_test,
                                                     cat_cols=[], n_splits=N_CV_SPLITS)
    # TODO: more tests
    assert len(X_test) <= TEST_SIZE * len(iris_data.data)
    assert len(X_train) <= TRAIN_SIZE * len(iris_data.data)
    assert len(y_test) <= TEST_SIZE * len(iris_data.data)
    assert len(y_train) <= TRAIN_SIZE * len(iris_data.data)
    assert (len(X_train) == len(y_train))
    assert (len(X_test) == len(y_test))
