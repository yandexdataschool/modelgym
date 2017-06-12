import numpy as np
import pickle, sys, argparse
from experiment import Experiment
from scipy.stats import t


class StatTest(object):

    def __init__(self, alpha=0.05, ttest_method='one-tailed'):
        self.alpha = alpha
        self.ttest_method = ttest_method
        

    def fit(self, first_losses, second_losses):
        assert first_losses.shape[0] == second_losses.shape[0], 'Lengths must be the same.'
        self.first_losses, self.second_losses = first_losses, second_losses
        self.size = self.first_losses.shape[0]
        self.diff = self.second_losses - self.first_losses
        
        self.first_mean, self.second_mean = self.first_losses.mean(), self.second_losses.mean()
        self.first_std, self.second_std = self.first_losses.std(), self.second_losses.std()
        self.diff_mean, self.diff_std = self.second_mean - self.first_mean, self.diff.std()
        assert self.diff_std > 0, 'Samples must be different'
        
        self.first_interval = self.student_conf_interval(self.first_mean, self.first_std, self.alpha)
        self.second_interval = self.student_conf_interval(self.second_mean, self.first_std, self.alpha)
        
        self.statistic, self.pvalue, self.ttest_interval = self.ttest(alpha=self.alpha,
                                                                      method=self.ttest_method)
        return self

        
    def student_conf_interval(self, mean, std, alpha=0.05, method='two-tailed'):
        if method == 'two-tailed':
            base_interval = t.ppf([alpha / 2, 1 - alpha / 2], self.size - 1) 
        elif method == 'one-tailed':
            base_interval = t.ppf([alpha, 1], self.size - 1)
        else: 
            assert False, 'Method must be "two-tailed" or "one-tailed"'
        interval = mean + base_interval * std / np.sqrt(self.size)
        return tuple(interval)
        
    
    def ttest(self, alpha=0.05, method='one-tailed'):
        statistic = self.diff_mean / self.diff_std * np.sqrt(self.size)
        pvalue = t.sf(np.abs(statistic), self.size - 1)
        if method == 'two-tailed': 
            pvalue = pvalue * 2
            
        interval = self.student_conf_interval(self.diff_mean, self.diff_std, alpha, method)
        return statistic, pvalue, interval


    def show(self):
        print 'Null hypothesis: \tmean1 == mean2'
        print 'Alternative hypothesis: mean1 %s mean2\n' % ('< ' if self.ttest_method == 'one-tailed' else '!=')
        for i, mean, std, interval in [('1', self.first_mean, self.first_std, self.first_interval),
                                       ('2', self.second_mean, self.second_std, self.second_interval)]:
            print 'mean{0}={1:.5f}\tstd{0}={2:.5f}\tinterval=({3:.5f}, {4:.5f})'.format(i, mean, std, 
                                                                                        interval[0], interval[1])
        print '\n%s paired t-test:' % self.ttest_method.capitalize()
        print '\tt-statistic = %.7f' % self.statistic
        print '\tpvalue      = %.7f' % self.pvalue
        print '\tt-confidence interval: (%.7f, %.7f)' % self.ttest_interval
        
    
    def dump(self, file_name):
        result = {
            'size': self.size, 'alpha': self.alpha, 'ttest_method': self.ttest_method,
            'diff': {'mean': self.first_mean, 'std': self.first_std},
            'first': {'mean': self.first_mean, 'std': self.first_std, 'interval': self.first_interval},
            'second': {'mean': self.second_mean, 'std': self.second_std, 'interval': self.second_interval},
            'ttest_result': {'statistic': self.statistic, 'pvalue': self.pvalue, 'interval': self.ttest_interval}
        }
        with open(file_name, 'wb') as f:
            pickle.dump(result, f)
        
    
    def load(self, file_name):
        with open(file_name, 'r') as f:
            result = pickle.load(f)

        self.size, self.alpha, self.ttest_method = result['size'], result['alpha'], result['ttest_method']
        self.first_mean, self.second_mean = result['first']['mean'], result['second']['mean']
        self.first_std, self.second_std = result['first']['std'], result['second']['std']
        self.first_interval, self.second_interval = result['first']['interval'], result['second']['interval']
        self.diff_mean, self.diff_std = result['diff']['mean'], result['diff']['std']
        
        self.statistic = result['ttest_result']['statistic']
        self.pvalue = result['ttest_result']['pvalue']
        self.ttest_interval = result['ttest_result']['interval']

        return self
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first_exp_result_path')
    parser.add_argument('-s', '--second_exp_result_path')
    parser.add_argument('-a', '--alpha', type=float, default=0.05)
    parser.add_argument('-m', '--ttest_method', choices=['one-tailed', 'two-tailed'], default='one-tailed')
    parser.add_argument('-o', '--output_file_name', default=None)
    namespace = parser.parse_args(sys.argv[1:])
        
    first_losses = Experiment().load(namespace.first_exp_result_path)[1]
    second_losses = Experiment().load(namespace.second_exp_result_path)[1]
            
    stattest = StatTest(namespace.alpha, namespace.ttest_method)
    stattest.fit(first_losses, second_losses)

    if namespace.output_file_name:
        stattest.show()

    if not namespace.output_file_name is None:
        stattest.dump(namespace.output_file_name)
        print 'Results are saved to %s' % namespace.output_file_name
        