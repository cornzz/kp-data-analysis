import numpy as np
import pandas as pd
import keras
from keras.datasets import boston_housing
from keras.backend import clear_session
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
from keras.utils import plot_model


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
    res = pd.DataFrame(index=range(1, 6), columns=[1024, 512, 256, 128, 64])
    for width in res.columns:
        for depth in res.index:
            print(f'Params: w_1={width}, l_max={depth}')
            # Create model with given params, train and evaluate
            clear_session()
            m = get_model(width, depth)
            m.fit(X_tr, y_tr, epochs=100, verbose=0)
            res[width][depth] = m.evaluate(X_te, y_te)

    return res


# Load training and test data
(X_train, y_train), (X_test, y_test) = boston_housing.load_data(seed=123)

# ---------------- Task 1 ----------------

model = get_model(512, 2)
model.fit(X_train, y_train, epochs=100, verbose=0)
print('------ Evaluation ------')
model.evaluate(X_test, y_test)
# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)

# ---------------- Task 2 ----------------
# Min-max normalize features
d_min, d_max = X_train.min(axis=0), X_train.max(axis=0)
X_train_norm = (X_train - d_min) / (d_max - d_min)
X_test_norm = (X_test - d_min) / (d_max - d_min)

model = get_model(512, 2)
model.fit(X_train_norm, y_train, epochs=100, verbose=0)
print('------ Evaluation normalized ------')
model.evaluate(X_test_norm, y_test)

# ---------------- Task 3 ----------------
# Grid search for best width/depth parameters
gs_res = grid_search(X_train_norm, y_train, X_test_norm, y_test)
best_w_1 = gs_res.columns[gs_res.min(axis=0).argmin()]
best_l_max = gs_res.index[gs_res.min(axis=1).argmin()]
print(f'Best params: w_1={best_w_1}, l_max={best_l_max}')
print(gs_res)

# ---------------- Grid search normalized ----------------
# Best params: w_1=1024, l_max=4
#       1024     512      256      128      64  
# 1   24.274  28.1363   30.283  35.1818  45.0328
# 2  14.7067  19.6051  17.2633   20.394  22.6159
# 3  16.1514  13.5092   14.558  17.0562  20.1662
# 4  11.3387  14.8458  17.1559  18.5711  19.8363
# 5  13.1483  15.0039  14.7446  16.6699  16.1235
