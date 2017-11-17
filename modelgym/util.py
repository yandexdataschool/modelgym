import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from skopt.space import Integer
from skopt.space import Real

from modelgym.cat_counter import CatCounter
from modelgym.model import TASK_CLASSIFICATION, TASK_REGRESSION
from modelgym.compare_auc_delong_xu import delong_roc_test


def preprocess_cat_cols(X_train, y_train, cat_cols, X_test=None, cc=None,
                        counters_sort_col=None, learning_task=TASK_CLASSIFICATION):
   if cc is None:
       sort_values = None if counters_sort_col is None else X_train[:, counters_sort_col]
       cc = CatCounter(learning_task, sort_values)
       X_train[:,cat_cols] = cc.fit(X_train[:,cat_cols], y_train)
   else:
       X_train[:,cat_cols] = cc.transform(X_train[:,cat_cols])
   if not X_test is None:
       X_test[:,cat_cols] = cc.transform(X_test[:,cat_cols])
   return cc


def elementwise_loss(y, p, learning_task=TASK_CLASSIFICATION):
    if learning_task == TASK_CLASSIFICATION:
        p_ = np.clip(p, 1e-16, 1 - 1e-16)
        return - y * np.log(p_) - (1 - y) * np.log(1 - p_)
    return (y - p) ** 2


def split_and_preprocess(X_train, y_train, X_test, y_test, cat_cols=[], n_splits=5, random_state=0, holdout_size=0,
                         learning_task=TASK_CLASSIFICATION):
    if holdout_size > 0:
        print('Holdout is used for counters.')
        X_train, X_hout, y_train, y_hout = train_test_split(X_train, y_train,
                                                            test_size=holdout_size,
                                                            random_state=random_state)
        cc = preprocess_cat_cols(X_hout, y_hout, cat_cols)
    else:
        cc = None

    CVSplit = KFold if learning_task == TASK_REGRESSION else StratifiedKFold
    cv = CVSplit(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_pairs = []
    for train_index, test_index in cv.split(X_train, y_train):
        fold_X_train = X_train[train_index]
        fold_X_test = X_train[test_index]
        fold_y_train = y_train[train_index]
        fold_y_test = y_train[test_index]
        preprocess_cat_cols(fold_X_train, fold_y_train, cat_cols, fold_X_test, cc)
        dtrain = XYCDataset(fold_X_train.astype(float), fold_y_train, cat_cols)
        dtest = XYCDataset(fold_X_test.astype(float), fold_y_test, cat_cols)
        cv_pairs.append((dtrain, dtest))

    _ = preprocess_cat_cols(X_train, y_train, cat_cols, X_test, cc)
    full_dtrain = XYCDataset(X_train.astype(float), y_train, cat_cols)
    full_dtest = XYCDataset(X_test.astype(float), y_test, cat_cols)

    return cv_pairs, (full_dtrain, full_dtest)


def hyperopt2skopt_space(space):
    global arrlist
    attr = ['loguniform', 'quniform', 'uniform', 'qloguniform']
    sw = ['float', 'switch']
    pardic = {}
    for parName in space:
        arrlist = []
        x1 = str(space.get(parName))
        # print("\t\torigin\n", x1)
        strings = x1.split()
        hpFunc = strings[1]
        if hpFunc == sw[0]:
            for word in strings:
                if attr.__contains__(word):
                    iter = 2
                    if word == attr[1] or word == attr[3]:
                        iter = 3
                    x1 = x1[x1.find(word):]
                    param = []
                    for i in range(0, iter):
                        x1 = x1[x1.find("{") + 1:]
                        idx = x1.find("}")
                        param.append(float(x1[:idx]))
                    # loguniform
                    if word == attr[0]:
                        pardic[parName] = Real(np.exp(param[0]), np.exp(param[1]))
                    # quiniform
                    elif word == attr[1]:
                        if param[2] == 1.0:
                            pardic[parName] = Integer(int(param[0]), int(param[1]))
                        else:
                            raise NotImplementedError()
                    # uniform
                    elif word == attr[2]:
                        pardic[parName] = Real(param[0], param[1], prior='uniform')
                    # qloguniform
                    elif word == attr[3]:
                        if param[2] == 1.0:
                            pardic[parName] = Integer(int(np.exp(param[0])), int(np.exp(param[1])))
                        else:
                            raise NotImplementedError()
                    else:
                        raise ValueError()
        elif hpFunc == sw[1]:
            for word in strings:
                if word == 'randint':
                    x1 = x1[x1.find(word):]
                    x1 = x1[len(word):]
                    dop = "Literal{"
                    x1 = x1[x1.find(dop):]
                    x1 = x1[len(dop):]
                    length = int(x1[:x1.find("}")])
                    for i in range(0, length):
                        x1 = x1[x1.find(dop):]
                        x1 = x1[len(dop):]
                        par = x1[:x1.find("}")]
                        try:
                            a = float(par)
                            tmp = int(a) - a
                            if tmp == 0:
                                a = int(a)
                            elif tmp == 0.0:
                                a = a
                        except:
                            a = par
                        arrlist.append(a)
                    break
            pardic[parName] = arrlist
        else:
            raise NotImplementedError()
    return pardic


class XYCDataset:
    def __init__(self, X, y, cat_cols):
        self.X = X
        self.y = y
        self.cat_cols = cat_cols

    def get_label(self):
        return self.y


def compare_models_different(first_model, second_model, data, alpha=0.05, metric='ROC_AUC'):
    """
    Hypothesis: two models are the same
    """
    if metric == 'ROC_AUC':
        X = data.X
        y_true = data.y
        p_value = 10**delong_roc_test(y_true, first_model.predict(X), second_model.predict(X))  # delong_roc_test returns log10(pvalue)
        if p_value < alpha:
            return True, p_value
        else:
            return False, p_value
    else:
        pass
