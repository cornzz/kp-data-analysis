import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sse(alpha, x, y):
    return np.sum((y - x @ alpha)**2)


def ols(alpha, x, y):
    return sse(alpha, x, y)


def ols_mat(x, y):
    x_t = np.transpose(x)
    y_t = np.transpose(y)
    return np.linalg.inv(x_t @ x) @ (x_t @ y_t)


def ridge(alpha, x, y, lam=1):
    return sse(alpha, x, y) + lam * np.sum(alpha ** 2)


def lasso(alpha, x, y, lam=1):
    return sse(alpha, x, y) + lam * np.sum(np.abs(alpha))


def elastic_net(alpha, x, y, lam_1=1, lam_2=1):
    return sse(alpha, x, y) + lam_1 * np.sum(alpha ** 2) + lam_2 * np.sum(np.abs(alpha))


# Read and unpack data, concat x column with ones for modeling the constant in later matrix multiplication
data_x, data_y = np.loadtxt('data-OLS.csv', dtype=float, delimiter=',', skiprows=1, unpack=True)
data_x = np.c_[data_x, np.ones(len(data_x))]

print('---- Task 1: OLS ----')
params_ols = minimize(ols, x0=[0, 0], args=(data_x, data_y)).x
print(f'y_i = {params_ols[0]} x_i + {params_ols[1]}')
print('SSE:', sse(params_ols, data_x, data_y))

print('\n---- Task 2: OLS matrix ----')
params_ols_mat = ols_mat(data_x, data_y)
print(f'y_i = {params_ols_mat[0]} x_i + {params_ols_mat[1]}')
print('SSE:', sse(params_ols_mat, data_x, data_y))

time_ols = timeit.timeit('minimize(ols, x0=[0, 0], args=(data_x, data_y))', number=10, globals=globals()) / 10
time_ols_mat = timeit.timeit('ols_mat(data_x, data_y)', number=10, globals=globals()) / 10
print(f'time_ols: {time_ols * 1000} ms')
print(f'time_ols_mat: {time_ols_mat * 1000} ms')

print('\n---- Task 3: ridge, lasso, elastic net regression ----')
params_ridge_reg = minimize(ridge, x0=[0, 0], args=(data_x, data_y, 9)).x
params_lasso_reg = minimize(lasso, x0=[0, 0], args=(data_x, data_y, 9)).x
params_elastic_net_reg = minimize(elastic_net, x0=[0, 0], args=(data_x, data_y, 4, 5)).x
print('Ridge regression:')
print(f'y_i = {params_ridge_reg[0]} x_i + {params_ridge_reg[1]}')
print('SSE:', sse(params_ridge_reg, data_x, data_y))
print('Lasso regression:')
print(f'y_i = {params_lasso_reg[0]} x_i + {params_lasso_reg[1]}')
print('SSE:', sse(params_lasso_reg, data_x, data_y))
print('Elastic net regression:')
print(f'y_i = {params_elastic_net_reg[0]} x_i + {params_elastic_net_reg[1]}')
print('SSE:', sse(params_elastic_net_reg, data_x, data_y))

plt.plot(data_x[:, 0], data_y, '.')
plt_x = np.linspace(0, 1, 2)
plt.plot(plt_x, params_ols[0]*plt_x + params_ols[1], 'r')
plt.plot(plt_x, params_ridge_reg[0]*plt_x + params_ridge_reg[1], 'g')
plt.plot(plt_x, params_lasso_reg[0]*plt_x + params_lasso_reg[1], 'b')
plt.plot(plt_x, params_elastic_net_reg[0]*plt_x + params_elastic_net_reg[1], 'y')
plt.legend(['data points', 'ols', 'ridge_reg', 'lasso_reg', 'elastic_net_reg'])
plt.show()
