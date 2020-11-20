import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data = np.loadtxt('cost.csv', delimiter=',', dtype=int)
rows = len(data)
cols = len(data[0])


def f(x, y=None):
    if y is None:
        return data[x[0], x[1]] if (-1 < x[0] < rows and -1 < x[1] < cols) else np.inf
    return data[x, y] if (-1 < x < rows and -1 < y < cols) else np.inf


def nelder_mead(points, alpha, gamma, rho, sigma):
    # Repeat while std of points costs > 2
    while np.std(points[:, 2]) > 2:
        # Sort points by cost
        points = points[points[:, 2].argsort()]
        # Calc centroid with best two points and reflection of worst point
        x_0 = points[:-1, :-1].mean(axis=0, dtype=int)

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(data, cmap='terrain_r', aspect='equal')
        # points_plot = points[:, :-1].copy()
        # triangle = plt.Polygon(np.flip(points_plot), fill=False, ec="r")
        # ax.add_patch(triangle)
        # ax.scatter([43], [144], c="r")
        # ax.plot()
        # plt.show()

        # Calc vector from worst point to centroid and reflection
        v = x_0 - points[-1, :-1]
        x_r = x_0 + alpha * v
        r_cost = f(x_r)
        if r_cost < points[0, -1]:
            # Expand if better than reflection, else reflect
            x_e = x_0 + gamma * v
            e_cost = f(x_e)
            if e_cost < r_cost:
                points[-1] = [*x_e, e_cost]
            else:
                points[-1] = [*x_r, r_cost]
        elif points[0, -1] <= r_cost < points[-2, -1]:
            # Reflect
            points[-1] = [*x_r, r_cost]
        else:
            # Contract if better than worst point, else shrink
            x_c = points[-1, :-1] + (rho * v).astype(int)
            c_cost = f(x_c)
            if c_cost < points[-1, -1]:
                points[-1] = [*x_c, c_cost]
            else:
                points[1:, :-1] = points[0, :-1] + (sigma * (points[1:, :-1] - points[0, :-1])).astype(int)
                points[1:, -1] = [f(point) for point in points[1:, :-1]]

    return points[0]


start = np.array([[rows - 1, 0, f(rows - 1, 0)],
                  [rows - 1, cols - 1, f(rows - 1, cols - 1)],
                  [0, cols // 2, f(0, cols // 2)]])
minimum_nm = nelder_mead(start, alpha=1, gamma=2, rho=0.5, sigma=0.5)
print(minimum_nm)
