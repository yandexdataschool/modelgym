import numpy as np
from sklearn.model_selection import KFold
from copy import copy

class XYCDataset:
    def __init__(self, X, y=None, cat_cols=[]):
        self.X = X
        self.y = y
        self.cat_cols = cat_cols
   
    def split(self, n_splits):
        """splits dataset into n_splits parts:
        
        Args:
            n_splits: int, number of splits to split dataset.
            
        Returns:
            splitted_dataset: list of XYCDataset object
        """
        
        indices = np.array_split(np.arange(self.y.shape[0]), n_splits)

        return [XYCDataset(self.X[indices_part], self.y[indices_part], 
                           copy(self.cat_cols)) for indices_part in indices]

    def cv_split(self, n_folds, random_state=None, shuffle=False):
        """return cross validation folds of dataset into n_folds

        Args:
            n_folds: number of folds
            random_state: random state
            shuffle (bool): whether to shuffle the data 
                before splitting into batches.
        Returns:
            list of tuples of 2 XYCDataset's: cross validation folds
        """
        cv = KFold(n_folds, random_state=random_state, shuffle=shuffle)
        cv_pairs = []
        for train_index, test_index in cv.split(self.X, self.y):
            fold_X_train = self.X[train_index]
            fold_X_test = self.X[test_index]
            fold_y_train = self.y[train_index]
            fold_y_test = self.y[test_index]
            dtrain = XYCDataset(fold_X_train, fold_y_train, copy(self.cat_cols))
            dtest = XYCDataset(fold_X_test, fold_y_test, copy(self.cat_cols))
            cv_pairs.append((dtrain, dtest))
        return cv_pairs

