import timeit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error as mse


def mse_arma(params, p, q, X):
    start = min(p, q)
    X_pred = run_arma(params[:p], params[p], params[p + 1:], p, q, X)

    return mse(X[start:], X_pred[start:])


def run_arma(phi, c, theta, p, q, X):
    epsilon, X_pred = [], []
    for t in range(len(X)):
        ar = sum([phi[i] * X[t - i - 1] for i in range(p)]) if t - p >= 0 else 0
        ma = sum([theta[i] * epsilon[t - i - 1] for i in range(q)]) if t - q >= 0 else 0
        x_t_pred = ar + ma + c
        X_pred.append(x_t_pred)
        epsilon.append(X[t] - x_t_pred if t > q - 1 else 0)

    return X_pred


data = pd.read_csv('ts_data.csv')
results = []

time1 = timeit.default_timer()

for shop_id in data['shop_id'].unique():
    X_ = data[data['shop_id'] == shop_id]['transactions'].values
    best_mse, best_params = np.inf, []

    for p_ in range(4):
        for q_ in range(4):
            params_ = np.zeros(p_ + 1 + q_)  # [phi, c, theta]
            res = opt.minimize(mse_arma, params_, args=(p_, q_, X_), method='Nelder-Mead')
            if res.fun < best_mse:
                best_mse = res.fun
                best_params = np.concatenate((res.x, [p_, q_]))

    p_, q_ = int(best_params[-2]), int(best_params[-1])
    phi_, c_, theta_ = best_params[:p_], best_params[p_], best_params[p_ + 1:-2]
    plt.plot(X_)
    plt.plot(run_arma(phi_, c_, theta_, p_, q_, X_))
    plt.title(f'Shop {shop_id}')
    plt.show()
    results.append([shop_id, p_, q_, phi_, c_, theta_, best_mse])

time2 = timeit.default_timer()

print(pd.DataFrame(results, columns=['shop_id', 'p', 'q', 'phi', 'c', 'theta', 'MSE']).set_index('shop_id').to_string())
print(f'Run time: {time2 - time1}')  # ~ 75s
