import datetime
import numpy as np
from networkx import Graph
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import *
from scipy.sparse import identity
from scipy import *
from sortedcontainers import SortedDict
from sortedcontainers import SortedSet
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import networkx as nx

def input_sets(location):
    data = np.genfromtxt(location, skip_header=1, dtype=int)
    G = dict()
    for c in data:
        x = c[0]
        y = c[1]
        if not y in G:
            G[y] = SortedSet()
        G[y].add(x)
    return G

def main():
    infilename = 'email-Eu-core-department-labels'
    outfilename = infilename + "_base"
    dir = 'testing/departments/'
    txt = '.txt'
    infilename = dir + infilename + txt
    outfilename = dir + outfilename + txt
    outfile = open(outfilename, 'w')

    sets = input_sets(infilename)

    for idx in sets:
        for i in range(len(sets[idx])):
            outfile.write(str(sets[idx][i]))
            if i < len(sets[idx]) - 1:
                outfile.write(' ')
        outfile.write('\n')

    outfile.close()
    return

if __name__ == '__main__':
    main()