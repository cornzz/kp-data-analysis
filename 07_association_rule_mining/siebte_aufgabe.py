import pandas as pd

from fileinput import input
from collections import defaultdict
from itertools import combinations

import timeit


# ------------------- Apriori algorithm -------------------


def get_frequent(items, tr, k, minsup):
    """
    Counts occurrences of given itemsets in transactions and eliminates sets that don't have required minimum support

    :param items:   List of itemsets
    :param tr:      List of transaction sets
    :param k:       Length of itemsets
    :param minsup:  Minimum support required for frequent itemsets
    :return:        Frequent itemsets
    """
    counts = []
    for i in items:
        for t in tr:
            if i <= t:
                counts.append([*i, 1])
    counts = pd.DataFrame(counts)
    counts = counts.groupby(list(range(k)), as_index=False).sum()
    return counts[counts.iloc[:, -1] >= minsup]


def apriori_gen(L, k):
    """
    Generates candidate supersets C of length k from frequent itemsets L

    :param L:   DataFrame where rows are frequent itemsets of length k-1, last column is frequency
    :param k:   Length of candidate itemsets of length k
    :return:    DataFrame where rows are candidate itemsets of length k
    """
    L = L[L.columns[:-1]]
    L['key'] = 1
    X = pd.merge(L, L, on='key').drop(columns=['key'])
    L.drop(columns='key', inplace=True)
    # Only keep combinations of sets with equal prefix
    C = X[(X[X.columns[:k-2]].values == X[X.columns[k-1:-1]].values).all(axis=1) &
          (X[X.columns[k-2]] < X[X.columns[-1]])].reset_index(drop=True)

    prune = []
    if k > 2:
        # Prune candidate itemsets that have subsets which are not in L
        L_sets = [set(row) for row in L.values]
        for i, c in enumerate([set(x) for x in C.values]):
            for subs in combinations(c, k-1):
                if set(subs) not in L_sets:
                    prune.append(i)
                    break

    return C.drop(index=prune)


def apriori(t, items, minsup):
    """
    Generates itemsets of frequent items by iteratively passing over the data while increasing the size of itemsets
    (Exponential complexity)

    :param t:       List of transactions (sets of items)
    :param items:   Set of all items that occur in the transactions
    :param minsup:  Minimum support (frequency) for frequent itemsets
    :return:        List of DataFrames of frequent itemsets
    """
    minsup = len(t) * minsup
    all_L = []
    # Get frequent single-itemsets
    L = get_frequent([{x} for x in items], t, 1, minsup)
    k = 2
    # Iteratively create candidate supersets C and prune non-frequent ones
    # until no further frequent supersets can be created
    while not L.empty:
        all_L.append(L.reset_index(drop=True))
        C = apriori_gen(L.copy(), k)
        if C.empty:
            break
        L = get_frequent([set(row) for row in C.values], t, k, minsup)
        k += 1

    return all_L


# ------------------- Eclat algorithm -------------------


def eclat(d, minsup):
    """
    Generates itemsets of frequent items by transforming to a vertical layout and forming set intersections
    (Equivalence CLAss Transformation)

    :param d:       List of transactions (sets of items)
    :param minsup:  Minimum support (frequency) for frequent itemsets
    :return:        List of dictionaries {itemset: transactions}
    """
    minsup = len(d) * minsup
    vertical = defaultdict(set)
    # Create vertical layout, columns are items, rows are transactions
    for (i, t) in enumerate(d):
        for item in t:
            vertical[item].add(i)

    all_L = []
    # Only keep columns with sufficient support
    L = {i: t for i, t in vertical.items() if len(t) >= minsup}
    k = 2
    # Iteratively create combinations of columns keeping the intersection of column content
    while len(L):
        all_L.append(L)
        vertical_new = {}
        for comb in combinations(L.keys(), 2):
            # Only combine columns with equal prefix
            if k > 2 and comb[0].split(' ')[:-1] != comb[1].split(' ')[:-1]:
                continue

            items = ' '.join(sorted(set(item for key in comb for item in key.split(' '))))
            vertical_new[items] = set.intersection(*[vertical[x] for x in comb])

        L = {i: t for i, t in vertical_new.items() if len(t) >= minsup}
        vertical = vertical_new
        k += 1

    return all_L


# ------------------- Task setup -------------------


test_data = [['items.tsv', 0.7], ['retail.tsv', 0.1]]
times = {'Algorithm': ['Apriori', 'Eclat']}

for test_set in test_data:
    transactions = [set(line.strip().split(' ')) for line in input(test_set[0])]
    all_items = set().union(*transactions)

    time1 = timeit.default_timer()
    task1 = apriori(transactions, all_items, test_set[1])
    time2 = timeit.default_timer()
    task2 = eclat(transactions, test_set[1])
    time3 = timeit.default_timer()
    times[test_set[0]] = ([time2 - time1, time3 - time2])

print(f'-------- Task 1: Apriori algorithm: --------')
for k, L in enumerate(task1):
    print(f'L{k+1}:\n{L.values}')

print(f'\n-------- Task 2: Eclat algorithm: --------')
for k, L in enumerate(task2):
    print(f'L{k+1}:')
    for s, t in L.items():
        print(f'{s} ({len(t)})')

print(f'\n-------- Task 3: Run times: --------')
print(pd.DataFrame(times).set_index('Algorithm'))

#            items.tsv  retail.tsv
# Algorithm                       
# Apriori     0.041602  145.219030
# Eclat       0.000219    0.351871
