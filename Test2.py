from collections import defaultdict
from math import sqrt
import random
import csv

def densify(x, n):
    """Convert a sparse vector to a dense one."""
    d = [0] * n
    for i, v in x:
        d[i] = v
    return d

def dist(x, c):
    """Euclidean distance between sample x and cluster center c.

    Inputs: x, a sparse vector
            c, a dense vector
    """
    sqdist = 0.
    for i, v in x:
        sqdist += (v - c[i]) ** 2
    return sqrt(sqdist)


def mean(xs, l):
    """Mean (as a dense vector) of a set of sparse vectors of length l."""
    c = [0.] * l
    n = 0
    for x in xs:
        for i, v in x:
            c[i] += v
        n += 1
    for i in range(l):
        c[i] /= n
    return c

def kmeans(k, xs, l, n_iter=10):
    # Initialize from random points.
    centers = [densify(xs[i], l) for i in random.sample(range(len(xs)), k)]
    cluster = [None] * len(xs)

    for _ in range(n_iter):
        for i, x in enumerate(xs):
            cluster[i] = min(range(k), key=lambda j: dist(xs[i], centers[j]))
        for j, c in enumerate(centers):
            members = (x for i, x in enumerate(xs) if cluster[i] == j)
            centers[j] = mean(members, l)

    return cluster


if __name__ == '__main__':
    # Cluster a bunch of text documents.
    import re
    import sys

    k = 3
    vocab = {}
    xs = []
    ns=[]
    cat=[]
    filename='2013-01.csv'
    with open(filename, newline='') as f:
        try:
            newsreader = csv.reader(f)
            for row in newsreader:
                ns.append(row[3])
                cat.append(row[4])
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, newsreader.line_num, e))

    k=3

    for a in ns:
       x = defaultdict(float)
       for w in re.findall(r"\w+", a):
                vocab.setdefault(w, len(vocab))
                x[vocab[w]] += 1
       xs.append(x.items())

    cluster_ind = kmeans(k, xs, len(vocab))
    clusters = [set() for _ in range(k)]
    for i, j in enumerate(cluster_ind):
        clusters[j].add(i)

    for j, c in enumerate(clusters):
        print("cluster %d:" % j)
        for i in c:
            print("\t%s" % cat[i])