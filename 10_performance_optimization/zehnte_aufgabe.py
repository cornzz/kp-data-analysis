import math
import timeit

import numpy as np


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def task1(n):
    code = '[dist(p1, p2) for p1 in data1 for p2 in data2]'
    return timeit.timeit(code, number=n, globals=globals()) / n


def task2(n):
    setup = """import multiprocessing as mp;\
    p = mp.Pool(mp.cpu_count())"""
    code = 'p.starmap(dist, [(p1, p2) for p1 in data1 for p2 in data2])'
    return timeit.timeit(code, setup, number=n, globals=globals()) / n


def task3(n):
    setup = 'from euclid import euclid_c'
    code = 'euclid_c(data1, data2)'
    return timeit.timeit(code, setup, number=n, globals=globals()) / n


def task4(n):
    setup = """d1 = np.repeat(data1, len(data2), axis=0);\
    d2 = np.tile(data2, (len(data1), 1))"""
    code = 'np.sqrt(np.sum((d1 - d2) ** 2, axis=1))'
    return timeit.timeit(code, setup, number=n, globals=globals()) / n


data1 = np.random.rand(2000, 2).tolist()
data2 = np.random.rand(2000, 2).tolist()
iterations = 10

if __name__ == '__main__':
    print(f'Task 1 - run time regular: \t {task1(iterations):.4f} s')       # ~ 1.29 s
    print(f'Task 2 - run time parallel: \t {task2(iterations):.4f} s')      # ~ 1.70 s
    print(f'Task 3 - run time C: \t\t {task3(iterations):.4f} s')           # ~ 0.70 s
    print(f'Task 4 - run time vectorized: \t {task4(iterations):.4f} s')    # ~ 0.17 s

