class XYCDataset:
    def __init__(self, X, y, cat_cols):
        self.X = X
        self.y = y
        self.cat_cols = cat_cols

    def get_label(self):
        return self.y;
