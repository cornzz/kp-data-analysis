import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


data = np.loadtxt('data-OLS.csv', dtype=float, delimiter=',', skiprows=1)


def ols(alpha):
    return np.sum([(p[1] - np.matmul([p[0], 1], alpha))**2 for p in data])


min_params = minimize(ols, x0=np.array([0, 0]))
print('------------------ Task 1 ------------------')
print(min_params.x, f'-> y_i = {min_params.x[0]} x_i + {min_params.x[1]}')

# plt.plot(data[:, 0], data[:, 1], 'o')
# x = np.linspace(0, 1, 2)
# plt.plot(x, min_params.x[0]*x + min_params.x[1], 'r')
# plt.show()


def ols_mat():
    x = np.c_[data[:, 0], np.ones(len(data))]
    y_t = np.transpose(data[:, 1])
    x_t = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(x_t, x)), np.matmul(x_t, y_t))


min_params = ols_mat()
print('------------------ Task 2 ------------------')
print(min_params, f'-> y_i = {min_params[0]} x_i + {min_params[1]}')

time_ols = np.mean(timeit.repeat('minimize(ols, x0=[0, 0])', 'from __main__ import ols, minimize', repeat=5, number=5)) / 5
print(time_ols)
time_ols_mat = np.mean(timeit.repeat('ols_mat()', 'from __main__ import ols_mat', repeat=5, number=5)) / 5
print(time_ols_mat)
