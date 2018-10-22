import numpy as np
import pandas as pd


class DummyMaker:
    """Class takes a categorical variable and returns a DataFrame with a column
    for each category having values of 0 or 1 for each row.
    """

    def __init__(self):
        self.levels = None

    def fit(self, categorical_array):
        """Array should take form df[col].
        u is list of unique categories.
        indices is list such that u[indices] will recreate the ordering of
        df[col].
        """
        u, indices = np.unique(categorical_array, return_inverse=True)
        self.levels = [u, indices]

    def transform(self, categorical_array, k_minus_one=False):
        """Array should be the same df[col] used in fit.
        k_minus_one not yet implemented.
        """
        num_rows = len(categorical_array)
        num_features = len(self.levels[0])
        dummies = np.zeros(shape=(num_rows, num_features))
        for idx, categ in enumerate(self.levels[0]):
            column = np.where(self.levels[1] == idx, 1, 0)
            dummies[:, idx:idx+1] += column.reshape(num_rows, 1)
        return pd.DataFrame(dummies, columns=self.levels[0])
