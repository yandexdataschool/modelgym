import pytest
from modelgym.utils import XYCDataset
import numpy as np

def test_cv_split():
    objects = 20
    features = 5
    X = np.arange(objects * features).reshape((objects, features)).astype(float)
    y = np.arange(objects)
    cat_cols = [1, 2]
    dataset = XYCDataset(X, y, cat_cols)
    cv_splits = dataset.cv_split(4)
    assert len(cv_splits) == 4
    for split in cv_splits:
        assert len(split) == 2
        assert split[0].cat_cols == cat_cols
        assert split[1].cat_cols == cat_cols
        assert len(split[0].X) == 15
        assert len(split[0].y) == 15
        assert len(split[1].X) == 5
        assert len(split[1].y) == 5
    
    for split in cv_splits:
        split[0].cat_cols[:] = []
        assert split[1].cat_cols != split[0].cat_cols
        assert split[0].cat_cols != cat_cols

def test_split():
    objects = 20
    features = 5
    X = np.arange(objects * features).reshape((objects, features)).astype(float)
    y = np.arange(objects)
    cat_cols = [1, 2]
    dataset = XYCDataset(X, y, cat_cols)
    splitted_dataset = dataset.split(6)
    assert len(splitted_dataset) == 6

    for i, ds in enumerate(splitted_dataset):
        if i < 2:
            assert len(ds.X) == 4
            assert len(ds.y) == 4
        else:
            assert len(ds.X) == 3
            assert len(ds.y) == 3

        assert ds.cat_cols == cat_cols
        ds.cat_cols[0] = 3
        assert ds.cat_cols != cat_cols