import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
from itertools import combinations


def plot_clusters(clust, c):
    for i in np.arange(len(clust)):
        cls = clust[i]
        plt.plot(data[cls, 0], data[cls, 1], 'o', c=np.random.rand(3))
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

clusters, centroids = k_means(7)
plot_clusters(clusters, centroids)


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
    # Create cluster array with indices of all data points
    clust = [[i] for i in np.arange(len(data))]
    # Iterate until desired amount of clusters reached
    while len(clust) != cut:
        # Calculate all combinations of clusters and min distance
        cl_prod = list(combinations(clust, 2))
        min_dist = np.argmin([calc_dist(data[x], data[y], fun) for x, y in cl_prod])
        cl1, cl2 = cl_prod[min_dist]
        # Add new cluster and remove old subclusters
        clust.append([*cl1, *cl2])
        clust.remove(cl1)
        clust.remove(cl2)

    return clust


# Reduce data set to get acceptable run time (/32 ~ 0.5s, /16 ~ 4.5s, /8 ~ 32.5s, /4 ~300s, /2 ~2580s)
# This is a straightforward, naive agnes implementation and thus not very efficient
data = data[np.arange(0, len(data), 16)]
time1 = time.time()
clusters = agnes(7, 'centroid')  # single / complete / average / centroid
print(f'Agnes: {time.time() - time1:.3f}s')
plot_clusters(clusters, None)
