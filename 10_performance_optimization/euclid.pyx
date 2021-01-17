import math


def euclid_c(list d1,list d2):
    dist = []
    for p1 in d1:
        for p2 in d2:
            dist.append(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    return dist
