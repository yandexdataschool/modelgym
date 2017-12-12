from modelgym.models import Model
from sklearn.metrics import accuracy_score
class Trainer(object):
    def __init__(self, model_spaces, tracker=None):
        raise NotImplementedError()

    def crossval_optimize_params(self, opt_metric, dataset, cv=3, 
                                 opt_evals=50, metrics=[], batch_size=10,
                                 verbose=False):
        raise NotImplementedError()

    def get_best_results():
        raise NotImplementedError()

    @staticmethod
    def eval_metrics(model_type, params, dtrain, dtest, metrics):
        assert(issubclass(model_type, Model))
        
        model = model_type(params=params)
        model.fit(dtrain)
        y_pred_proba = None
        y_pred = None
        
        if any(map(lambda metric: metric.requires_proba, metrics)):
            y_pred_proba = model.predict_proba(dtest)
        if not all(map(lambda metric: metric.requires_proba, metrics)):
            y_pred = model.predict(dtest)
        
        metrics_results = {}
        for metric in metrics:
            pred = y_pred_proba if metric.requires_proba else y_pred
            metrics_results[metric.name] = metric.calculate(dtest.y, pred)

        return metrics_results

    