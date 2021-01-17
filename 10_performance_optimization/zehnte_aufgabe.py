import math
import timeit

import numpy as np


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def task1():
    setup = """from __main__ import dist, data1, data2;\
    import math"""
    code = '[dist(p1, p2) for p1 in data1 for p2 in data2]'
    return timeit.timeit(code, setup, number=1)


def task2():
    setup = """from __main__ import dist, data1, data2;\
    import multiprocessing as mp;\
    p = mp.Pool(mp.cpu_count())"""
    code = 'p.starmap(dist, [(p1, p2) for p1 in data1 for p2 in data2])'
    return timeit.timeit(code, setup, number=1)


def task3():
    return timeit.timeit('euclid_c(data1, data2)', 'from __main__ import data1, data2; from euclid import euclid_c', number=1)


data1 = np.random.rand(2000, 2).tolist()
data2 = np.random.rand(2000, 2).tolist()

if __name__ == '__main__':
    print(f'Task 1 - run time euclid: {task1()} s')
    print(f'Task 2 - run time euclid parallel: {task2()} s')
    print(f'Task 3 - run time euclid C: {task3()} s')

