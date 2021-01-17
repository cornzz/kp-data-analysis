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
        ar = sum([phi[i] * X[t - i - 1] for i in range(p)]) if t >= p else 0
        ma = sum([theta[i] * epsilon[t - i - 1] for i in range(q)]) if t >= q else 0
        x_t_pred = ar + ma + c
        X_pred.append(x_t_pred)
        epsilon.append(X[t] - x_t_pred if t >= q else 0)

    start = min(p, q)
    return [x if i >= start else np.nan for i, x in enumerate(X_pred)]


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
print(f'Run time: {time2 - time1}')  # ~ 74s

#          p  q                                                               phi           c                                                             theta           MSE
# shop_id
# 145      2  2                          [1.577945890243841, -0.5805208567562562]    0.111205                         [-1.197286120657726, 0.10304867129026313]    215.534145
# 260      0  3                                                                []   80.883642      [0.2700625607034346, 0.251368703108013, 0.36902996690161805]    358.991219
# 315      2  2                         [0.15121227453070596, 0.8587056038264328]   -0.599630                         [0.2435650184400094, -0.7450204256811005]    323.665112
# 375      1  3                                              [0.9460710622804878]    2.508986   [-0.7200216007615525, -0.0785587156357711, -0.3846864663817191]    229.710109
# 548      3  3  [-0.005177916974552462, 0.15808890890164007, 0.8353133125654155]    0.660439   [0.03498124550944211, -0.2575389121649492, -1.0969139247802944]    335.365771
# 560      3  3   [0.8503698807213711, 0.42865143342907497, -0.27594663022412846]   -0.623565   [-0.5165049672769322, -0.8367293980187558, 0.18358899910286935]   1076.115160
# 750      3  3    [1.137177647568584, -0.41880113171381983, 0.27862219429677204]    0.619535  [-1.0252674083409408, 0.39340038667564753, -0.22829493663712963]    567.198041
# 897      2  0                        [0.48766996086408865, -0.5294570934787806]  408.807478                                                                []  11017.815856
# 1332     2  3                        [1.2264626999510813, -0.23596949356948732]    0.614065   [-0.8029995153965732, -0.4089639293630146, 0.06080798225313451]   3299.616715
# 1750     2  2                         [1.5099210677672286, -0.5037642773577748]   -0.323318                        [-1.0631066333638826, -0.0611620443647589]    429.402128
