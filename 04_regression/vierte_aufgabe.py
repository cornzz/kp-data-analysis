import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Read and unpack data, concat x column with ones for modeling the constant in later matrix multiplication
data_x, data_y = np.loadtxt('data-OLS.csv', dtype=float, delimiter=',', skiprows=1, unpack=True)
data_x = np.c_[data_x, np.ones(len(data_x))]


def sse(alpha):
    return np.sum((data_y - np.matmul(data_x, alpha))**2)


def ols():
    return minimize(sse, x0=np.array([0, 0]))


params_ols = ols()
print('------------------ Task 1 ------------------')
print(params_ols.x, f'-> y_i = {params_ols.x[0]} x_i + {params_ols.x[1]}')
print('SSE:', sse(params_ols.x))


def ols_mat():
    x_t = np.transpose(data_x)
    y_t = np.transpose(data_y)
    return np.matmul(np.linalg.inv(np.matmul(x_t, data_x)), np.matmul(x_t, y_t))


params_ols_mat = ols_mat()
print('------------------ Task 2 ------------------')
print(params_ols_mat, f'-> y_i = {params_ols_mat[0]} x_i + {params_ols_mat[1]}')
print('SSE:', sse(params_ols_mat))

time_ols = np.mean(timeit.repeat('ols()', 'from __main__ import ols, minimize', repeat=5, number=5)) / 5
time_ols_mat = np.mean(timeit.repeat('ols_mat()', 'from __main__ import ols_mat', repeat=5, number=5)) / 5
print(f'time_ols: {time_ols * 1000} ms')
print(f'time_ols_mat: {time_ols_mat * 1000} ms')


def ridge_reg(lam=1):
    return minimize(lambda alpha: sse(alpha) + lam * np.sum(alpha ** 2), x0=np.array([0, 0]))


def lasso_reg(lam=1):
    return minimize(lambda alpha: sse(alpha) + lam * np.sum(np.abs(alpha)), x0=np.array([0, 0]))


def elastic_net_reg(lam_1=1, lam_2=1):
    return minimize(lambda alpha: sse(alpha) + lam_1 * np.sum(alpha ** 2) + lam_2 * np.sum(np.abs(alpha)),
                    x0=np.array([0, 0]))


params_ridge_reg = ridge_reg(9)
params_lasso_reg = lasso_reg(9)
params_elastic_net_reg = elastic_net_reg(4, 5)
print('------------------ Task 3 ------------------')
print('Ridge regression:')
print(params_ridge_reg.x, f'-> y_i = {params_ridge_reg.x[0]} x_i + {params_ridge_reg.x[1]}')
print('SSE:', sse(params_ridge_reg.x))
print('Lasso regression:')
print(params_lasso_reg.x, f'-> y_i = {params_lasso_reg.x[0]} x_i + {params_lasso_reg.x[1]}')
print('SSE:', sse(params_lasso_reg.x))
print('Elastic net regression:')
print(params_elastic_net_reg.x, f'-> y_i = {params_elastic_net_reg.x[0]} x_i + {params_elastic_net_reg.x[1]}')
print('SSE:', sse(params_elastic_net_reg.x))

plt.plot(data_x[:, 0], data_y, '.')
plt_x = np.linspace(0, 1, 2)
plt.plot(plt_x, params_ols.x[0]*plt_x + params_ols.x[1], 'r')
plt.plot(plt_x, params_ridge_reg.x[0]*plt_x + params_ridge_reg.x[1], 'g')
plt.plot(plt_x, params_lasso_reg.x[0]*plt_x + params_lasso_reg.x[1], 'b')
plt.plot(plt_x, params_elastic_net_reg.x[0]*plt_x + params_elastic_net_reg.x[1], 'y')
plt.legend(['data points', 'ols', 'ridge_reg', 'lasso_reg', 'elastic_net_reg'])
plt.show()
