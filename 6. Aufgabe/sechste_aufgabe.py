import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time


def plot_clusters(clust, c):
    clr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in np.arange(len(clust)):
        cls = clust[i]
        plt.plot(data[cls, 0], data[cls, 1], 'o', c=clr[i % 9])
    if c is not None:
        plt.plot(c[:, 0], c[:, 1], 'X', c='r')
    plt.show()


def k_means(k):
    # Init cluster centroids randomly
    max_v = min(max(data[:, 0]), max(data[:, 1]))
    c = np.random.uniform(0, max_v, [k, 2]).round(2)
    c_p, clust = None, None

    while not np.array_equal(c, c_p):
        c_p = c.copy()
        # Reset clusters
        clust = [[] for i in np.arange(k)]
        # Iterate over all data points to calculate new clusters
        for i in np.arange(len(data)):
            # Calculate distance from this point to all centroids
            dist = [np.linalg.norm(data[i] - x) for x in c]
            # Find index of centroid with lowest distance
            min_idx = np.argmin(dist)
            # Add index of this point to selected cluster centroid
            clust[min_idx].append(i)
        # Iterate over all centroids to calculate new positions
        for i in np.arange(len(c)):
            if not len(clust[i]) == 0:
                c[i] = np.mean(data[clust[i]], axis=0)

    return clust, c


data = np.loadtxt('data-clust.csv', delimiter=',', skiprows=1)

# clusters, centroids = k_means(7)
# plot_clusters(clusters, centroids)

# ---- Task 2 ----


def calc_dist(cl1, cl2, fun):
    if fun != 'centroid':
        dist = [np.linalg.norm(x - y) for x, y in product(cl1, cl2)]
        return min(dist) if fun == 'single' \
            else max(dist) if fun == 'complete' \
            else np.sum(dist) / (len(cl1) * len(cl2))
    else:
        c_1, c_2 = np.mean(cl1, axis=0), np.mean(cl2, axis=0)
        return np.linalg.norm(c_1 - c_2)


def agnes(cut, fun):
    clust = [[i] for i in np.arange(len(data))]

    while len(clust) != cut:
        plot_clusters(clust, None)
        clust_new = []
        while len(clust) > 1 and len(clust) + len(clust_new) != cut:
            cl = clust.pop(0)
            min_dist = np.argmin([calc_dist(data[cl], data[x], fun) for x in clust])
            clust_new.append([*cl, *clust[min_dist]])
            del clust[min_dist]
        if len(clust) != 0:
            for cl in clust:
                clust_new.append(cl)
        clust = clust_new
    return clust


data = data[np.arange(0, len(data), 8)]

time1 = time.time()
clusters = agnes(7, 'average')  # single / complete / average / centroid
print(time.time() - time1)
print(clusters)
plot_clusters(clusters, None)
