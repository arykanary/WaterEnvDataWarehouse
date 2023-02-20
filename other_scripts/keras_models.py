import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

from common import create_set, split_train_val_test, sklearn_fit_eval


def train_keras(layers: list,
                x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray,
                y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                epoch: int, batch: int):
    model = Sequential([Input((x_tr.shape[1],)), *layers])
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.build()
    model.summary()
    model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch)

    return (model,
            (mean_squared_error(y_train, model.predict(x_train)),
             mean_squared_error(y_val, model.predict(x_val)),
             mean_squared_error(y_test, model.predict(x_test))))


df = pd.read_csv('../RWS/_data/input_data.csv').astype(float)
df = df[['Year', 'Month', 'Day', 'Hour', 'ArnhemWaterHoogte']]#.iloc[:60000]
print(df)
print(df.describe())

x_tr, x_va, x_te, y_tr, y_va, y_te = split_train_val_test(*create_set(df, 'ArnhemWaterHoogte', 8, 1, 0, True))
m, (train_mse, val_mse, test_mse) = train_keras([Dense(1000), Dense(1000), Dense(1000), Dense(1)],
                                                x_tr, x_va, x_te, y_tr, y_va, y_te,
                                                10, 3000)
print(train_mse, val_mse, test_mse)
