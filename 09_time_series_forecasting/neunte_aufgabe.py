import timeit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error as mse


def mse_arma(params, p, q, X):
    start = p + q
    X_pred = run_arma(params[:p], params[p], params[p + 1:], p, q, X)

    return mse(X[start:], X_pred[start:])


def run_arma(phi, c, theta, p, q, X):
    epsilon, X_pred = [], []
    for t in range(len(X)):
        ar = sum([phi[i] * X[t - i - 1] for i in range(p)]) if t >= p else X[t]
        ma = sum([theta[i] * epsilon[t - i - 1] for i in range(q)]) if t >= q else 0
        x_t_pred = ar + ma + c
        X_pred.append(x_t_pred)
        epsilon.append(X[t] - x_t_pred if t >= q else 0)

    return X_pred


def grid_search_arma(p_max, q_max, X):
    best_mse, best_params = np.inf, []

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            # Initialize starting parameters with 0
            # and find optimal parameters for given p, q
            params = np.zeros(p + 1 + q)  # [phi, c, theta]
            res = opt.minimize(mse_arma, params, args=(p, q, X), method='Nelder-Mead')
            # Replace best_params if mse of current params is smaller
            if res.fun < best_mse:
                best_mse = res.fun
                best_params = np.concatenate((res.x, [p, q]))

    return best_mse, best_params  # (mse, [[phi], c, [theta], p, q])


data = pd.read_csv('ts_data.csv')
results = []

time1 = timeit.default_timer()
# Iterate through shops
for shop_id in data['shop_id'].unique():
    # Select shop data and run grid search for best arma parameters
    X_ = data[data['shop_id'] == shop_id]['transactions'].values
    best_mse_, best_params_ = grid_search_arma(3, 3, X_)

    # Extract params from best_params_ and append to results
    p_, q_ = int(best_params_[-2]), int(best_params_[-1])
    phi_, c_, theta_ = best_params_[:p_], best_params_[p_], best_params_[p_ + 1:-2]
    results.append([shop_id, p_, q_, phi_, c_, theta_, best_mse_])

    # Plot time series forecast
    plt.plot(X_)
    plt.plot(run_arma(phi_, c_, theta_, p_, q_, X_))
    plt.title(f'Shop {shop_id}')
    plt.legend(['Original time series', 'Forecast'])
    plt.show()

time2 = timeit.default_timer()

print(pd.DataFrame(results, columns=['shop_id', 'p', 'q', 'phi', 'c', 'theta', 'MSE']).set_index('shop_id').to_string())
print(f'Run time: {time2 - time1}')  # ~ 78s

#          p  q                                                              phi          c                                                               theta          MSE
# shop_id
# 145      3  1  [1.2300560180781368, -0.43398203132955826, 0.15699685098188673]   3.846286                                               [-0.8142657889970104]   232.946975
# 260      0  3                                                               []  82.928032       [0.2677333394263928, 0.24441633254665399, 0.3631678517388479]   352.027256
# 315      3  3    [0.7908508383103495, -0.5192212515160447, 0.7246411112278135]   1.145741      [-0.6198382821879214, 0.6491465018686285, -0.5491356842000084]   304.152828
# 375      2  1                       [1.2252216105870997, -0.23437877755174202]   0.579817                                               [-1.1156840115097655]   232.746598
# 548      1  3                                             [0.9883296272506913]   1.150844  [-0.9794091145028707, -0.11730210951438347, -0.038995326981426465]   338.078882
# 560      2  3                        [0.9421534014728543, 0.05538736596783225]   0.744495   [-0.7898886076945599, -0.24287640900879492, -0.12325820990713474]  1137.284990
# 750      3  3    [0.14361997372441615, 0.5303758890360735, 0.3295626381852996]  -0.131812   [0.04800523547012561, -0.45015720918776236, -0.28755851203958815]   560.662778
# 897      3  1    [1.5321840884158493, -1.1338431097165813, 0.5793167028090167]  11.375154                                               [-1.1201533429906636]  9862.666387
# 1332     3  2  [1.0171727833884923, 0.02644156409842971, -0.05298650943003405]   0.609043                          [-0.8665870493982677, -0.3114553010192853]  3389.664878
# 1750     2  2                       [0.9999319977857655, 0.025148656394023902]  -3.413355                         [-0.6523904948048657, -0.47514978463822244]   421.358821
