import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, log_loss

class Metric(object):
    def __init__(self, scoring_function, requires_proba=False, is_min_optimal=False, name="default_name"):
        self._name = name
        self._scoring_function = scoring_function
        self._requires_proba = requires_proba
        self._is_min_optimal = is_min_optimal

    def get_predictions(self, model, bst, _dtest, dtest):
        prediction_func = model.predict_proba if self._requires_proba else model.predict
        prediction = prediction_func(bst, _dtest, dtest.X)
        if not self._requires_proba:
            prediction = np.round(prediction).astype(int)
        return prediction

    def calculate(self, model, bst, dtest, _dtest, sample_weight=None): # TODO: why 2 dtests??
        """Calculates required prediction from the model

        Depending on the metric we might need either of the following methods:
        .predict
        .predict_proba

        Args:
            model:  modelgym Model object, required to calculate predictions
            bst:    actual model to pass to modelgym Model
            dtest:  raw dataset
            _dtest: dataset converted by model
        Returns:
            score:  float, the result of the metric calculation 
        """

        prediction = self.get_predictions(model, bst, _dtest, dtest)

        score = self._scoring_function(_dtest.get_label(), prediction, 
                                       sample_weight=sample_weight) # TODO weights
        return score

    @property
    def is_min_optimal(self):
        return self._is_min_optimal
   
    @property
    def requires_proba(self):
        return self._requires_proba

    @property
    def name(self):
        return self._name

class RocAuc(Metric):
    def __init__(self, name='roc_auc'):
        super(RocAuc, self).__init__(scoring_function=roc_auc_score, name=name, requires_proba=True)

class Accuracy(Metric):
    def __init__(self, name='accuracy'):
        super(Accuracy, self).__init__(scoring_function=accuracy_score, name=name)

class F1(Metric):
    def __init__(self, name='f1_score'):
        super(F1, self).__init__(scoring_function=f1_score, name=name)

class Recall(Metric):
    def __init__(self, name='recall'):
        super(Recall, self).__init__(scoring_function=recall_score, name=name)

class Precision(Metric):
    def __init__(self, name='precision'):
        super(Precision, self).__init__(scoring_function=precision_score, name=name)

class Logloss(Metric):
    def __init__(self, name='logloss'):
        super(Logloss, self).__init__(scoring_function=log_loss, requires_proba=True, 
                                      is_min_optimal=True, name=name)
