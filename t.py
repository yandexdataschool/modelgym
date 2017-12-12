import modelgym as mg
from collections import namedtuple
from sklearn.datasets import make_classification

import hyperopt
import pickle
import numpy as np

class ModelSpace:
    def __init__(self, model_type, model_space):
        self.model_type = model_type
        self.model_space = model_space

params = mg.models.XGBClassifier.get_default_parameter_space()

params["objective"] = 'multi:softmax'
params["num_class"] = 5

# boston = load_boston()
# X = boston["data"]
# y = boston["target"]

sps = ModelSpace(mg.models.XGBClassifier, params)


# def read_data(fname, nrows=None, shuffle=True):
#     with open(fname,'rb') as fh:
#         X, y = pickle.load(fh,encoding='bytes')
#     index = np.arange(X.shape[0])
#     if nrows is None:
#         nrows = X.shape[0]
#     weights = np.ones(nrows) # uh, well...
#     if shuffle:
#         index_perm = np.random.permutation(index)
#     else:
#         index_perm = index
#     return X[index_perm[:nrows]], y[index_perm[:nrows]], weights

# X, y, weights = read_data("data/XY2d.pickle", nrows=2000)

# from xgboost import XGBClassifier

# model = XGBClassifier(n_estimators=1000)

# model.fit(X[:1800], y[:1800])
# pred = model.predict(X[1800:])
# from sklearn.metrics import accuracy_score

# print(accuracy_score(np.round(pred).astype(int), y[1800:]))

# exit(0)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_classes=5)

print(X[:10])

print(y[:10])

input()

ds = mg.utils.XYCDataset(X, y, None)

print(ds)

trainer = mg.HyperoptTrainer([sps], hyperopt.rand.suggest)
trainer.crossval_optimize_params(mg.metric.Accuracy(), ds)
print(trainer.get_best_results())