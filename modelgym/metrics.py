from sklearn.metrics import roc_auc_score, accuracy_score, \
                            f1_score, recall_score, precision_score, log_loss, \
                            mean_squared_error


class Metric(object):
    """Metric class is a wrapper around sklearn.metrics class, with additional information: when optimizing this
    metric, should we minimize it (like log_loss) or maximize (like accuracy), and whether it's calculation requires
    computed probabilities (like roc_auc).

    Of course, not only sklearn.metrics could be wrapped into this class
    """

    def __init__(self, scoring_function, requires_proba=False, is_min_optimal=False, name="default_name"):
        """
        Args:
            scoring_function (types.FunctionType): wrapped scoring function
            requires_proba (bool): whether calculation of metric requires computed probabilities
            is_min_optimal (bool): is the less the better
            name (str): name of metric
        """

        self._name = name
        self._scoring_function = scoring_function
        self._requires_proba = requires_proba
        self._is_min_optimal = is_min_optimal

    def calculate(self, y, y_pred):
        """Calculates metric from true labels and predictions

        Args:
            y: true labels
            y_pred: predicted labels or probabilities
        Returns:
            float: the result of the metric calculation
        """
        score = self._scoring_function(y, y_pred) # TODO weights
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


class Mse(Metric):
    def __init__(self, name='mse'):
        super(Mse, self).__init__(scoring_function=mean_squared_error, name=name)
