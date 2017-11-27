#-------------------------------------------------------------------------------
# Name:        algo
# Purpose:
#
# Author:      Milos
#
# Created:     26/11/2017
# Copyright:   (c) Milos 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
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

def get_beta(Qe, dinC, cardC, k):
    top = 0
    bot = 0
    for Ci in range(k):
        tmp = (dinC[Ci]*cardC[Ci])
        top += tmp * tmp
        bot += tmp
    bot = bot * bot
    return (bot*(1 - Qe) - top)/(bot * Qe + top)

def main():
    # mysmall example#
    # k = 4                 # number of communities
    # cardC = [5, 5, 5, 5]  # community sizes
    # dinC = [8, 7, 6, 7]   # internal degree of each community
    # doutC = []      # external degree of each community
    # N = 0           # total number of vertices
    # Qe = 0.2        # wanted modularity - determines the value of beta

    # if the graph is homogenious
    N = 10000
    gamma = 0.5
    k = int(pow(N, gamma))
    alpha = 4
    cardC = []
    dinC = []
    doutC = []
    Qe = 0.8

    for i in range(k):
        cardC.append(int(N/k))
        dinC.append(cardC[i] * int(alpha*log(cardC[i]))/2)

    idx_lbo = [0] # the lower bound for indices in each cluster
    for i in range(1,k):
        idx_lbo.append(idx_lbo[i-1]+cardC[i])

    N = 0
    for i in range(k):
        N += cardC[i]

    beta = get_beta(Qe, dinC, cardC, k)

    for i in range(k):
        doutC.append(int(dinC[i]*beta))

    dir = 'out/'
    txt = '.txt'
    outfilebase = '%s_gen' % '{:%H-%M-%S}'.format(datetime.datetime.now())
    outfilepath = dir + outfilebase + '_graph' + txt
    outfile = open(outfilepath, 'w')
    outfile.write("# k = %s\n" % k)
    outfile.write("# N = %s\n" % N)
    outfile.write("# alpha = %s\n" % alpha)
    outfile.write("# beta = %s\n" % beta)
    outfile.write("# gamma = %s\n" % gamma)
    outfile.write("# Qe = %s\n" % Qe)

    # determine all the internal edges
    lef = 0
    rig = 0
    all_edges = []
    for Ci in range(k):
        lef = rig
        rig += cardC[Ci]

        all_curr_edges = []
        for i in range(lef, rig-1):
            for j in range(i+1, rig):
                all_curr_edges.append((i,j))

        curr_selected_edges = []
        all_curr_edge_cnt = len(all_curr_edges)
        if dinC[Ci] < all_curr_edge_cnt:
            curr_selected_indices = np.random.choice(range(all_curr_edge_cnt), dinC[Ci], replace=False)
            for i in curr_selected_indices:
                curr_selected_edges.append(all_curr_edges[i])
        else:
            curr_selected_edges = all_curr_edges

        for i in range(len(curr_selected_edges)):
            all_edges.append(curr_selected_edges[i])

    # now on to the external ones
    remaining_doutC = dict()
    for i in range(k):
        remaining_doutC[i] = doutC[i]
    inter_connections = SortedSet()
    for C1 in range(k-1):
        for C2 in range(C1+1, k):
             inter_connections.add((C1, C2))

    # determine the "bundles" of edges between two clusters
    external_edge_bundle = dict()
    while remaining_doutC and len(inter_connections) > 0:
        curr_cnt = len(inter_connections)
        choice = np.random.choice(range(curr_cnt))
        (C1, C2) = inter_connections[choice]

        max_alloc = minimum(remaining_doutC[C1], remaining_doutC[C2])
        alloc = np.random.choice(range(1, max_alloc + 1))

        if (C1, C2) not in external_edge_bundle:
            external_edge_bundle[(C1, C2)] = 0
        external_edge_bundle[(C1, C2)] += alloc

        options = [C1,C2]
        for C in options:
            remaining_doutC[C] -= alloc
            if remaining_doutC[C] == 0:
                del remaining_doutC[C]
                for otherC in range(k):
                    pair = (minimum(C, otherC), maximum(C, otherC))
                    if pair in inter_connections:
                        inter_connections.remove(pair)

    for (C1, C2) in external_edge_bundle:
        all_curr_edges = []
        for i in range(idx_lbo[C1], idx_lbo[C1] + cardC[C1]):
            for j in range(idx_lbo[C2], idx_lbo[C2] + cardC[C2]):
                all_curr_edges.append((i, j))

        allowance = external_edge_bundle[(C1, C2)]
        max_edges = len(all_curr_edges)
        curr_selected_indices = np.random.choice(range(max_edges), allowance, replace=False)

        curr_selected_edges = []
        for i in curr_selected_indices:
            curr_selected_edges.append(all_curr_edges[i])

        for i in range(len(curr_selected_edges)):
            all_edges.append(curr_selected_edges[i])

    #G = Graph()
    #for (v1, v2) in all_edges:
    #    G.add_edge(v1, v2)
    #nx.draw(G)
    #fig = plt.figure(figsize=(25, 10))
    #G#
    #plt.draw()

    outfile.write("#\tFromNodeId\tToNodeId")
    outfile.flush()
    all_edges.sort()
    for (v1, v2) in all_edges:
        outfile.write("%s\t%s\n" % (v1, v2))
        outfile.flush()
    outfile.close()

    outfilepath = dir + outfilebase + '_calc' + txt
    outfile = open(outfilepath, 'w')
    for i in range(k):
        for j in range(idx_lbo[i], idx_lbo[i] + cardC[i]):
            if j > idx_lbo[i]:
                outfile.write(" ")
            outfile.write("%s" % j)
        outfile.write("\n")
        outfile.flush()
    outfile.close()
    return

if __name__ == '__main__':
    main()