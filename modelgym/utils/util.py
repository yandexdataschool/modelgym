import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from skopt.space import Integer
from skopt.space import Real

# from cat_counter import CatCounter
from modelgym.utils.compare_auc_delong_xu import delong_roc_test
from modelgym.metrics import Metric

def compare_models_different(first_model, second_model, data, alpha=0.05, metric='ROC_AUC'):
    """
    Hypothesis: two models are the same
    """
    if metric == 'ROC_AUC':
        X = data.X
        y_true = data.y
        p_value = 10 ** delong_roc_test(y_true, first_model.predict(X), second_model.predict(X))  # delong_roc_test returns log10(pvalue)
        if p_value < alpha:
            return True, p_value
        else:
            return False, p_value
    else:
        pass