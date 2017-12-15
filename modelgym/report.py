import seaborn
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

from modelgym.metrics import Metric


class Report:
    
    def __init__(self, results, test_set, metrics):
        self._results = results
        self._test_set = test_set
        self._metric_results = {metric: self._parse_results(metric.name) 
                                for metric in metrics}
        
    """  
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
                                  """
   
    def _parse_results(self, field):
        result = []
       
        for model_result in self._results.values():
            result.append(np.array([score[field] for score in model_result['result']['metric_cv_results']]).mean())     
        return np.array(result)
    
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
        formatted = ['{:.6f} ({:+.2f}%)'.format(self._metric_results[metric][i], diffs[i]) 
                     for i in range(len(self._results.keys()))]
        print (pd.DataFrame(formatted, columns=['tuned'], index=self._results.keys()))

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
        
        # stack position to sorting them and plotting afterwards
        number_of_models = len(self._results.keys())
        
        pyplot.figure(figsize=(20, 7))
        xticks = []
        for i, (model_name, model_result) in enumerate(self._results.items()):
            values = self._metric_results[metric][i]
            cv_results = np.array([score[metric.name] for score in model_result['result']['metric_cv_results']])
            errors = np.fabs(np.array([cv_results.min(), cv_results.max()]).reshape(2, -1) - values)
            pyplot.errorbar(i, values, yerr=errors, fmt='.', markersize=25, capsize=10, capthick=2, barsabove=True)
            xticks.append(model_name + ' tuned\n%.5f' % self._metric_results[metric][i])
            
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
        for model_name, model_class in self._results.items():
            model = model_class['model_space'].model_class(params=model_class['result']['params'])
            names.append(model_name)
            train_set, for_corr_set = self._test_set.split(2)
            model.fit(train_set)
            predictions.append(model.predict_proba(for_corr_set))

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
