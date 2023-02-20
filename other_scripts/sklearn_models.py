import pandas as pd
from typing import Union, Tuple
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from common import create_set, split_train_val_test, sklearn_fit_eval

model_hint = Union[KNeighborsRegressor, LinearRegression, Ridge, Lasso, ElasticNet,
                   DecisionTreeRegressor, RandomForestRegressor,
                   SVR]


def k_neighbor_variation(max_neighbors: int, data: pd.DataFrame, **kwargs):
    x, y = create_set(data, **kwargs)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y, **kwargs)
    n_dict = {}
    for k in range(1, max_neighbors):
        # k = 2**k
        print('k:', k)
        train_mse, val_mse, test_mse = sklearn_fit_eval(KNeighborsRegressor(k),
                                                        x_train, x_val, x_test, y_train, y_val, y_test)
        n_dict[k] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


def set_size_variation(model: model_hint, max_size: int, step_size: int, data: pd.DataFrame, **kwargs):
    n_dict = {}
    for n in range(step_size, max_size+step_size, step_size):
        print('size: ', n)
        x, y = create_set(data.iloc[:n], **kwargs)
        train_mse, val_mse, test_mse = sklearn_fit_eval(model(), *split_train_val_test(x, y, **kwargs))
        n_dict[n] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


def poly_variation(model: model_hint, max_poly: int, data: pd.DataFrame, **kwargs):
    n_dict = {}
    for n in range(max_poly):
        print('poly: ', n)
        x, y = create_set(data, 'ArnhemWaterHoogte', 1, 1, n, True)
        train_mse, val_mse, test_mse = sklearn_fit_eval(model(), *split_train_val_test(x, y, **kwargs))
        n_dict[n] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


def feature_variation(model: model_hint, data: pd.DataFrame, **kwargs):
    n_dict = {}
    x, y = create_set(data, **kwargs)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y, **kwargs)

    for n in range(1, x_train.shape[1]):
        print('features: ', n)
        train_mse, val_mse, test_mse = sklearn_fit_eval(model(), x_train[:, :n], x_val[:, :n], x_test[:, :n],
                                                        y_train, y_val, y_test)
        n_dict[n] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


def regularization_variation(model: Union[Ridge, Lasso], regularization: Tuple[float], data: pd.DataFrame, **kwargs):
    n_dict = {}
    x, y = create_set(data, **kwargs)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y, **kwargs)

    for n in range(*regularization):
        n /= 100000
        print('regularization: ', n)
        train_mse, val_mse, test_mse = sklearn_fit_eval(model(n), x_train, x_val, x_test,
                                                        y_train, y_val, y_test)
        n_dict[n] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


def back_variation(model: model_hint, back_max, data: pd.DataFrame, **kwargs):
    n_dict = {}

    for n in range(1, back_max):
        print('back: ', n)
        kwargs['back'] = n
        x, y = create_set(data, **kwargs)
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y, **kwargs)
        print(x_train.shape)
        train_mse, val_mse, test_mse = sklearn_fit_eval(model(), x_train, x_val, x_test,
                                                        y_train, y_val, y_test)
        n_dict[n] = (train_mse, val_mse)

    plt.plot(n_dict.keys(), n_dict.values())
    return n_dict


df = pd.read_csv('_data/dataset_prep.csv', index_col=0, parse_dates=True).astype(float)
# print(df.describe())

# k_neighbor_variation(20, df, back=100, shift=10, interaction=3,
#                      flat=True, target_name='Meetwaarde.Waarde_Numeriek', split=(.6, .2, .2), seed=1)

# regularization_variation(Lasso, (1000, 1000000, 1000), df,
#                          target_name='ARNH', back=2, shift=1, interaction=0, flat=True,
#                          split=(.6, .2, .2), seed=1)
# back_variation(LinearRegression, 20, df,
#                target_name='ArnhemWaterHoogte', back=2, shift=1, interaction=1, flat=True,
#                split=(.6, .2, .2), seed=1)
plt.show()

# ---single model training---
# lrm = LinearRegression()
# x_tr, x_va, x_te, y_tr, y_va, y_te = split_train_val_test(*create_set(df, 'ARNH', 8, 1, 0, True),
#                                                           (0.75, 0.2, .05))
# print(sklearn_fit_eval(lrm, x_tr, x_va, x_te, y_tr, y_va, y_te))
# print(lrm.predict(x_va), y_va)
#
knm = KNeighborsRegressor(15)  # 10 < k < 30
print(sklearn_fit_eval(knm, *split_train_val_test(*create_set(df, 'Meetwaarde.Waarde_Numeriek', 100, 10, 4, True))))

