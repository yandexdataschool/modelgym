import seaborn
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

from modelgym.metrics import Metric
from modelgym.util import calculate_custom_metrics


class Report:
    def __init__(self, results, models_dict, models_holder, test_set, metrics, task_type):
        self._models_dict = models_dict
        self._test_set = test_set
        self._task = task_type
        self._models_holder = models_holder
        self._metric_results = {metric: self._parse_results(results,
                                                            metric.name) 
                                  for metric in metrics}
        self._metric_bounds = {metric: self._parse_results(results,
                                                           metric.name + '_bounds') 
                                  for metric in metrics}
   
    def _parse_results(self, results, field):
        return np.array([(results[i]['default'][field], results[i]['tuned'][field])
                         for i in self._models_dict.keys()])
   
    def _calc_differences(self, metric):
        baseline = self._metric_results[metric].max() if metric.is_min_optimal else\
                   self._metric_results[metric].min()
        diffs = 100 * self._metric_results[metric] / baseline - 100
        return diffs
    
    def _print_metric_name(self, name):
        # TODO refactor this to utils and reuse in example.ipynb
        space_count = 4
        line_length = 105
        tilda_length = (line_length - len(name) - space_count) // 2
        print ("\n" + "~" * tilda_length + " " * space_count + name + " " * space_count +\
               "~" * (tilda_length if (line_length - len(name) - space_count) % 2 == 0 else tilda_length + 1) + "\n")
        
    def _print_differences_dataframe(self, metric, diffs):
        formatted = [['{:.6f} ({:+.2f}%)'.format(self._metric_results[metric][i, j], 
                                                 diffs[i, j]) for j in range(2)] 
                              for i in range(len(self._models_dict.keys()))]
        print (pd.DataFrame(formatted, columns=['default', 'tuned'], index=self._models_dict.keys()))

    def _validate_metric(self, metric):
        if metric not in self._metric_results:
            raise KeyError("Results for {} not found".format(metric.name))
        if not isinstance(metric, Metric):
            raise ValueError("metric must be a Metric class, found {}".format(type(metric)))
    
    def _print_metric_results(self, metric):
        """internal function to exclude meta information about report task going on"""
        diffs = self._calc_differences(metric)
        self._print_differences_dataframe(metric, diffs)
        
    def print_metric_results(self, metric):
        self._validate_metric(metric)
        self._print_metric_name(metric.name)
        self._print_metric_results(metric)
        
    def print_all_metric_results(self):
        for metric in self._metric_results.keys():
            self.print_metric_results(metric)
        
    def _plot_metric_results(self, metric):
        """internal function to exclude meta information about report task going on"""
        cartesian = lambda x, y: np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        
        # stack position to sorting them and plotting afterwards
        number_of_models = len(self._models_dict.keys())

        tiled_models = cartesian(np.arange(2), np.arange(number_of_models))[:, ::-1]
  
        sorted_tiled_models = np.array(sorted(tiled_models, 
                                           key=lambda row: self._metric_results[metric][row[0]][row[1]],
                                           reverse=not metric.is_min_optimal))
        pyplot.figure(figsize=(20, 7))
        for i in np.arange(number_of_models):
            positions = np.where(sorted_tiled_models[:, 0] == i)[0]
            models_indices = sorted_tiled_models[positions][:, 1]
            values = self._metric_results[metric][i][models_indices]
            errors = self._metric_bounds[metric][i][models_indices].reshape(2, -1)
            pyplot.errorbar(positions, values, yerr=errors, fmt='.', markersize=25, capsize=10, capthick=2, barsabove=True)
        
        modes = np.array(['default', 'tuned'])
        xticks = [list(self._models_dict.keys())[row[0]] + ' ' + modes[row[1]] +\
                 '\n%.5f' % self._metric_results[metric][row[0]][row[1]]
                            for row in sorted_tiled_models]
        
        pyplot.xticks(range(len(xticks)), xticks, fontsize=15)
        pyplot.yticks(fontsize=12)
        pyplot.title('Comparison', fontsize=20)
        pyplot.ylabel(metric.name, fontsize=16)
        pyplot.show()
        
    def plot_metric_results(self, metric):
        self._validate_metric(metric)
        self._print_metric_name(metric.name)
        self._plot_metric_results(metric)

    def plot_all_metrics(self):
        for metric in self._metric_results.keys():
            self.plot_metric_results(metric)

    def summary(self):
        for metric in self._metric_results.keys():
            self.print_metric_results(metric)
            self._plot_metric_results(metric)
            self.plot_heatmap(metric)
    
    def _plot_heatmap(self, metric):
        """internal function to exclude meta information about report task going on"""
        self._validate_metric(metric)
        predictions = []
        names = []
        for i, model_class in self._models_dict.items():
            model = model_class(self._task)
            name = model.get_name()
            names.append(name)
            estimator = self._models_holder[name].state['tuned_test']['bst']
            _dtest = model.convert_to_dataset(self._test_set.X, self._test_set.y, self._test_set.cat_cols)
            predictions.append(metric.get_predictions(model, estimator, _dtest, self._test_set))

        seaborn.set_style('ticks')
        fig, ax = pyplot.subplots()
        fig.set_size_inches(8, 8)
        fig.suptitle(metric.name)
        corr = np.corrcoef(np.array(predictions))
        seaborn.heatmap(corr, annot=True, xticklabels=names, yticklabels=names)
        pyplot.show()
        
    def plot_heatmap(self, metric):
        self._print_metric_name(metric.name)
        self._plot_heatmap(metric)

    def plot_heatmaps(self):
        for metric in self._metric_results.keys():
            self.plot_heatmap(metric)

    def print_metric_name(name):
        space_count = 4
        line_length = 110
        tilda_length = (line_length - len(name) - space_count) // 2
        print ("\n" + "~" * tilda_length + " " * space_count + name + " " * space_count +\
               "~" * (tilda_length if (line_length - len(name) - space_count) % 2 == 0 else tilda_length + 1) + "\n")
