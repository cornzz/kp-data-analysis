import numpy as np
import pandas as pd
from keras.datasets import boston_housing
from keras import Input
from keras.models import Sequential
from keras.layers import Dense


def get_model(w_1, l_max):
    m = Sequential()
    # Input layer
    m.add(Input(shape=(13,)))
    # Hidden layers
    for i in range(0, l_max):
        m.add(Dense(w_1, activation='relu'))
        w_1 /= 2
    # Output layer
    m.add(Dense(1))
    m.compile(optimizer='adam', loss='mean_squared_error')

    return m


def grid_search(X_tr, y_tr, X_te, y_te):
    res = pd.DataFrame(index=range(2, 8), columns=[1024, 512, 256, 128, 64, 32])
    best_loss, best_w, best_d = np.inf, 0, 0
    for width in res.columns:
        for depth in res.index:
            print(f'Params: w_1={width}, l_max={depth}')
            m = get_model(width, depth)
            m.fit(X_tr, y_tr, epochs=100, verbose=0)
            loss = m.evaluate(X_te, y_te)
            res[width][depth] = loss
            if loss < best_loss:
                best_loss, best_w, best_d = loss, width, depth

    return res, [best_w, best_d]


def normalize(data):
    data = data.copy()
    for i in range(len(data[0])):
        col = data[:, i]
        data[:, i] = (col - col.min()) / (col.max() - col.min())

    return data


(X_train, y_train), (X_test, y_test) = boston_housing.load_data(seed=123)

# ---------------- Task 1 ----------------

model = get_model(512, 2)
model.fit(X_train, y_train, epochs=100, verbose=0)
print('------ Evaluation ------')
model.evaluate(X_test, y_test)
# model.summary()
# keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# ---------------- Task 2 ----------------

X_train_norm, X_test_norm = normalize(X_train), normalize(X_test)

model = get_model(512, 2)
model.fit(X_train_norm, y_train, epochs=100, verbose=0)
print('------ Evaluation normalized ------')
model.evaluate(X_test_norm, y_test)

# ---------------- Task 3 ----------------

gs_res, best_params = grid_search(X_train_norm, y_train, X_test_norm, y_test)
print(f'Best params: w_1={best_params[0]}, l_max={best_params[1]}')
print(gs_res)

# ---------------- Grid search normalized ----------------
# Best params: w_1=1024, l_max=6
#       1024     512      256      128      64       32
# 2  14.8887  16.8898  18.1341  16.9749  22.4089  26.7596
# 3  16.2622  15.1979  16.1548  17.2807   20.384  24.8819
# 4  16.0572   15.964  17.3328    17.06  18.3923  30.3079
# 5  14.4549   14.572   14.993  17.0658  559.226  25.3321
# 6  13.4848  16.1995  19.3459  18.0536  20.1741  25.6963
# 7  18.3374  15.3112  18.1424  559.232  559.225  616.591
