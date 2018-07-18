import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

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

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )