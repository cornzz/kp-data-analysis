import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv('diabetes.csv')
mse = {'': ['MSE OLS regression',
            'MSE LASSO regression',
            'MSE OLS regression, after CBFS',
            'MSE OLS regression, after PCA'],
       'Train set': [],
       'Test set': []}

# ------------------- Task 1 - OLS regression -------------------

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

reg_OLS = LinearRegression().fit(X_train, Y_train)
Y_pred_train_OLS = reg_OLS.predict(X_train)
Y_pred_test_OLS = reg_OLS.predict(X_test)
mse_pred_train_OLS = mean_squared_error(Y_train, Y_pred_train_OLS)
mse_pred_test_OLS = mean_squared_error(Y_test, Y_pred_test_OLS)

# ------------------- Task 2 - LASSO regression -------------------

reg_LASS = Lasso(alpha=0.1).fit(X_train, Y_train)
Y_pred_train_LASS = reg_LASS.predict(X_train)
Y_pred_test_LASS = reg_LASS.predict(X_test)
mse_pred_train_LASS = mean_squared_error(Y_train, Y_pred_train_LASS)
mse_pred_test_LASS = mean_squared_error(Y_test, Y_pred_test_LASS)

# ------------------- Task 3 - Correlation-based Feature Selection -------------------

features = X_train.transpose().to_numpy()
corr_coef = np.corrcoef(features)
corr_f = [c for c in combinations(range(len(corr_coef)), 2) if abs(corr_coef[c]) > 0.6]
drop = []

while corr_f:
    counts = np.unique(corr_f, return_counts=True)
    d = counts[0][np.argmax(counts[1])]
    drop.append(d)
    corr_f = [x for x in corr_f if d not in x]

X_train_CBFS = X_train.drop(X_train.columns[drop], axis=1)
X_test_CBFS = X_test.drop(X_test.columns[drop], axis=1)

reg_OLS_CBFS = LinearRegression().fit(X_train_CBFS, Y_train)
Y_pred_train_OLS_CBFS = reg_OLS_CBFS.predict(X_train_CBFS)
Y_pred_test_OLS_CBFS = reg_OLS_CBFS.predict(X_test_CBFS)
mse_pred_train_OLS_CBFS = mean_squared_error(Y_train, Y_pred_train_OLS_CBFS)
mse_pred_test_OLS_CBFS = mean_squared_error(Y_test, Y_pred_test_OLS_CBFS)

# ------------------- Task 4 - Principal Component Analysis -------------------

XX = X_train.T @ X_train
_, eig_v = np.linalg.eig(XX)
W = np.transpose(eig_v)
W_2 = W[:, :2]
X_train_PCA = X_train @ W_2

X_test_PCA = X_test.drop(X_test.columns[2:], axis=1)

reg_OLS_PCA = LinearRegression().fit(X_train_PCA, Y_train)
Y_pred_train_OLS_PCA = reg_OLS_PCA.predict(X_train_PCA)
Y_pred_test_OLS_PCA = reg_OLS_PCA.predict(X_test_PCA)
mse_pred_train_OLS_PCA = mean_squared_error(Y_train, Y_pred_train_OLS_PCA)
mse_pred_test_OLS_PCA = mean_squared_error(Y_test, Y_pred_test_OLS_PCA)

# ------------------- Result printing -------------------

mse['Train set'] = [mse_pred_train_OLS, mse_pred_train_LASS, mse_pred_train_OLS_CBFS, mse_pred_train_OLS_PCA]
mse['Test set'] = [mse_pred_test_OLS, mse_pred_test_LASS, mse_pred_test_OLS_CBFS, mse_pred_test_OLS_PCA]

print(pd.DataFrame(mse).set_index(''))
