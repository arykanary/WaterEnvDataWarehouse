import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class WindowGenerator:
    """"""
    def __init__(self, data: pd.DataFrame, name, distribution=(3, 1, 2), split=(.7, .2, .1), squeeze=False, delete_target=True, shuffle=False, normalizer=MinMaxScaler):
        if shuffle:
            self.input_data = data.sample(frac=1)
        else:
            self.input_data = data

        self.col_num = {n: v for v, n in enumerate(data.columns)}
        
        if isinstance(name, (str, int)):
            self.col_ind = self.col_num[name]
        elif isinstance(name, (tuple, list)):
            self.col_ind = [self.col_num[x] for x in name]
        elif name is None:
            self.col_ind = [self.col_num[x] for x in self.input_data.columns]

        
        if isinstance(normalizer, bool) and normalizer:
            self._after_norm = True
        elif normalizer is not None and not isinstance(normalizer, bool):
            self.normalizer = normalizer()
            self.input_data = self.normalizer.fit_transform(self.input_data)
        self.input_data = self.input_data.astype(np.float32)

        self.squeeze = squeeze
        self.delete_target = delete_target

        assert round(sum(split), 4) == 1.
        self.train_size, self.valid_size, _ = split

        self.distribution = sum(distribution)
        self.back, self.shift, self.fore = distribution
        self.row, self.col = self.input_data.shape
        
        # Init others
        self.window_data = None
        self.features, self.targets = None, None
        self.train, self.valid, self.testd = (None, None), (None, None), (None, None)
        
    def window(self):
        """"""
        repeat = np.repeat(np.expand_dims(self.input_data, axis=0), self.distribution, axis=0)
        ind = np.expand_dims(np.array([list(range(x, (self.row - self.distribution+x+1))) for x in range(self.distribution)]), 2)
        self.window_data = np.stack(np.take_along_axis(repeat, ind, axis=1), axis=1)
    
    def get_target(self):
        """"""
        self.targets   = self.window_data[:, -self.fore:, self.col_ind]
        if self.delete_target:
            self.features = np.delete(self.window_data, self.col_ind, 2)[:, :self.back]
        else:
            self.features = self.window_data[:, :self.back]
    
    def split(self):
        """"""
        self.train_size = int(self.train_size * self.row)
        self.valid_size = int(self.valid_size * self.row) + self.train_size
        
        self.train = (self.features[:self.train_size], self.targets[:self.train_size])
        self.valid = (self.features[self.train_size:self.valid_size], self.targets[self.train_size:self.valid_size])
        self.testd = (self.features[self.valid_size:], self.targets[self.valid_size:])

        if self.squeeze:
            trainx, trainy = self.train
            self.train = np.squeeze(trainx), np.squeeze(trainy)
            validx, validy = self.valid
            self.valid = np.squeeze(validx), np.squeeze(validy)
            testdx, testdy = self.testd
            self.testd = np.squeeze(testdx), np.squeeze(testdy)
    
    def normalize(self):
        f_tr, t_tr = self.train
        f_va, t_va = self.valid
        f_te, t_te = self.testd

        self.feature_mean = np.mean(f_tr, axis=(0, 1))
        self.feature_std  = np.std(f_tr, axis=(0, 1))
        self.target_mean = np.mean(t_tr)
        self.target_std  = np.std(t_tr)
        
        self.train = (f_tr - self.feature_mean) / self.feature_std, (t_tr - self.target_mean) / self.target_std
        self.valid = (f_va - self.feature_mean) / self.feature_std, (t_va - self.target_mean) / self.target_std
        self.testd = (f_te - self.feature_mean) / self.feature_std, (t_te - self.target_mean) / self.target_std

    def __call__(self):
        """"""
        self.window()
        self.get_target()
        self.split()
        if self._after_norm:
            self.normalize()


def test_wg():
    # Test data
    df = pd.DataFrame(np.arange(21).reshape(7, 3))
    wg_window = np.array([[[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11], [12, 13, 14], [15, 16, 17]],
                                              [[ 3,   4,   5], [ 6,   7,   8], [ 9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]])

    wg_features = np.array([[[ 0,  2],  [ 3,  5],   [ 6,  8]], [[ 3,  5],  [ 6,   8],  [ 9, 11]]])

    wg_targets = np.array([[13, 16], [16, 19]])
    
    wg_train_x, wg_train_y = np.array([[[ 0,  2], [ 3,  5],  [ 6,  8]], [[ 3,  5], [ 6,  8], [ 9, 11]]]), np.array([[13, 16], [16, 19]])
    
    # Testing
    wg = WindowGenerator(df)
    wg.window()
    assert np.array_equal(wg.window_data, wg_window)

    wg.get_target(1)
    assert np.array_equal(wg.features, wg_features)
    assert np.array_equal(wg.targets, wg_targets)

    wg.split()
    x, y = wg.train
    assert np.array_equal(x, wg_train_x)
    assert np.array_equal(y, wg_train_y)
    
    print(wg.valid)
    print(wg.testd)
    
    print('All succes!')
