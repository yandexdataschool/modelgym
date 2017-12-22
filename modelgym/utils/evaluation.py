import numpy as np

def crossval_fit_eval(model_type, params, cv, metrics, verbose):
    """Run cross validation of model and calculate metric results
    Args:
        model_type (type, subclass of Model)
        params (dict of str:obj): model parameters
        cv (list of tuple like (XYCDataset, XYCDataset)): cross validation folds
        metrics (list of modelgym.metrics.Metric): metrics to evaluate.
            Last metric is considered to be either loss (if metric.is_min_optimal is True) or -loss.
            Loss is the metric we want to minimize.
        verbose (bool): Enable verbose output.
    Returns:
        dict of shape: {
            "loss": float,
            "metric_cv_results": list,
            "params": dict
        },
        metric_cv_results contains dict's from metric names to calculated metric values for each fold in cv_fold
        params is just a copy of input argument params
    """
    metric_cv_results = []
    losses = []
    for dtrain, dtest in cv:
        eval_result = \
            eval_metrics(model_type, params, dtrain, dtest, metrics)
        metric_cv_results.append(eval_result)

        if metrics[-1].is_min_optimal:
            loss = eval_result[metrics[-1].name]
        else:
            loss = -eval_result[metrics[-1].name]
        losses.append(loss)

    return {
        "loss": np.mean(losses),
        "metric_cv_results": metric_cv_results,
        "params": params.copy(),
    }


def eval_metrics(model_type, params, dtrain, dtest, metrics):
    """Evaluates model with given parameters on train dataset,

    evaluates on test dataset and calculates given metrics

    Args:
        model_type (type, subclass of modelgym.models.Model): class of model
            (e.g modelgym.models.XGBClassifier)
        params (dict of str: obj): model parameters
        dtrain (modelgym.utils.XYCDataset): train dataset
        dtest (modelgym.utils.XYCDataset): test dataset
        metrics (list of modelgym.metrics.Metric): metrics to evaluate
    Returns:
        dict of str: obj: Mapping from metric name to metric result
    """

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
