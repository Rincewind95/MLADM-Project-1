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

import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import *
from scipy.sparse import identity
from scipy import *
from sortedcontainers import SortedDict
from sortedcontainers import SortedSet
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Merging Dendogram')
        plt.xlabel('cluster elements')
        plt.ylabel('joined clusters')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def load_graph(location):
    data = np.genfromtxt(location, skip_header=1, dtype=int)
    G = dict()
    for c in data:
        x = c[0]-1
        y = c[1]-1
        if not x in G:
            G[x] = []
        G[x].append(y)
    return G


def exp_squaring(P, t, node_cnt):
    R = np.identity(node_cnt, dtype='float')
    if t == 0:
        return R
    else:
        while t > 1:
            if t % 2 == 0:
                P = P * P
                t = t / 2
            else:
                R = P * R
                P = P * P
                t = (t - 1) / 2
        return P * R

def get_P_t(G, t):
    # artificially modify G to have 'self-loops'
    maxidx = 0
    Gnew = dict()
    for v1 in G.keys():
        if v1 > maxidx:
            maxidx = v1
        added = False
        Gnew[v1] = []
        for v2 in G[v1]:
            if v2 > maxidx:
                maxidx = v2
            if v2 == v1:
                added = True
            elif v2 > v1:
                if not added:
                    Gnew[v1].append(v1)
                    added = True
            Gnew[v1].append(v2)
        if not added:
            Gnew[v1].append(v1)
    maxidx = maxidx + 1
    indices = []
    indexptr = [0]
    data = []
    for curr in range(maxidx):
        if not curr in Gnew.keys():
            indices.append(0)
            data.append(0.0)
            indexptr.append(len(indices))
            continue
        neig_cnt = len(Gnew[curr])
        for neig in Gnew[curr]:
            indices.append(neig)
            data.append(1.0/neig_cnt)
        indexptr.append(len(indices))

    P = csr_matrix((data, indices, indexptr), dtype=float)
    P = exp_squaring(P, t, maxidx)
    return (P, maxidx)

def get_r2_C1C2(D, P_t_C1, P_t_C2):
    r = D * P_t_C1 - D * P_t_C2
    res = 0
    for elem in r:
        res += elem*elem
    return res

def delta_sigma_C1C2(n, cardC1, cardC2, D, P_t_C1, P_t_C2):
    r2_C1C2 = get_r2_C1C2(D, P_t_C1, P_t_C2)
    return ((cardC1 * cardC2) / ((cardC1 + cardC2)*n))*r2_C1C2

def delta_sigma_C3C(n, cardC1, cardC2, cardC, D, P_t_C1, P_t_C2, P_t_C, deltaC1C2):
    C1C = (cardC1 + cardC)*delta_sigma_C1C2(n, cardC1, cardC, D, P_t_C1, P_t_C)
    C2C = (cardC2 + cardC)*delta_sigma_C1C2(n, cardC2, cardC, D, P_t_C2, P_t_C)
    C1C2 = cardC*deltaC1C2
    return (C1C + C2C - C1C2)/(cardC1 + cardC2 + cardC)


def remove_elem(sigma_to_C1C2, C1C2_to_sigma, oldkey):
    sigmaC1C2 = C1C2_to_sigma[oldkey]
    del C1C2_to_sigma[oldkey]
    sigma_to_C1C2[sigmaC1C2].remove(oldkey)
    if len(sigma_to_C1C2[sigmaC1C2]) <= 0:
        del sigma_to_C1C2[sigmaC1C2]
    return


def add_elem(sigma_to_C1C2, C1C2_to_sigma, newkey, newsigma):
    C1C2_to_sigma[newkey] = newsigma
    if not newsigma in sigma_to_C1C2:
        sigma_to_C1C2[newsigma] = SortedSet()
    sigma_to_C1C2[newsigma].add(newkey)
    return

def main():
    G = load_graph('example.txt')
# G = load_graph('../resources/facebook/facebook_combined.txt')
   # G = load_graph('andrej_test2.txt')

    t = 3                            # choice of t
    (P_t, n) = get_P_t(G, t)         # the Pt matrix from the question set and the total number of vertices in the graph
    Deg = dict()                     # degree of each vertex
    D = []                           # diagonal matrix of vertex degrees ^(-1/2)
    Ccard = dict()                   # the mapping of community indices to their respective cardinality
    CP_t = dict()                    # the mapping of community indices to their respective P_t_C_.
    Cneig = dict()                   # the mapping of community indices to their "neighbour" communities (ones who share a direct edge to them)
    C1C2_to_sigma = dict()           # the mapping of two adjacent community indices to their respective sigma (assumed to be unique enough!)
    unsorted_sigma_to_C1C2 = dict()  # a mapping of sigmas (updated with new values regularly)
    merge_order = []                 # a list of tuples in set merge order ((C1,C2) means C1 and C2 were merged to C1)

    for v in G.keys():
        Ccard[v] = 1.0
    for v in G.keys():
        Deg[v] = len(G[v])
    for v in Deg.keys():
        D.append(1.0 / sqrt(Deg[v]+1))
    for v in G.keys():
        Cneig[v] = set(G[v])
    for v in G.keys():
        CP_t[v] = P_t[v, :]
    for C1 in G.keys():
        for C2 in G[C1]:
            if not (C1, C2) in C1C2_to_sigma and C1 < C2:
                sigmaC1C2 = delta_sigma_C1C2(n, Ccard[C1], Ccard[C2], D, CP_t[C1], CP_t[C2])
                C1C2_to_sigma[(C1, C2)] = sigmaC1C2
                if sigmaC1C2 not in unsorted_sigma_to_C1C2:
                    unsorted_sigma_to_C1C2[sigmaC1C2] = SortedSet()
                unsorted_sigma_to_C1C2[sigmaC1C2].add((C1, C2))

    sigma_to_C1C2 = SortedDict(unsorted_sigma_to_C1C2) # a constantly sorted mapping of sigmas (updated with new values regularly)

    # iterate through all merges
    for _ in range(n-1):
        # select the minimum element in the sorted set, record and remove itit
        (C1, C2) = sigma_to_C1C2.viewvalues()[0][0]
        sigmaC1C2 = C1C2_to_sigma[(C1,C2)]
        merge_order.append((C1, C2, sigmaC1C2))
        del C1C2_to_sigma[(C1,C2)]
        del sigma_to_C1C2[sigmaC1C2][0]
        if len(sigma_to_C1C2[sigmaC1C2]) <= 0:
            del sigma_to_C1C2[sigmaC1C2]

        # calculate the values for the new community
        P_t_C3 = (Ccard[C1]*CP_t[C1] + Ccard[C2]*CP_t[C2])/(Ccard[C1] + Ccard[C2])
        cardC3 = Ccard[C1] + Ccard[C2]

        # determine which tuples need updating and resolve maintenance things
        updatePoints = [C1, C2]
        updatePairs = set()
        for A in updatePoints:
            neighbours = Cneig[A]
            for B in neighbours:
                oldkey = (A, B)
                if A > B:
                    oldkey = (B, A)
                if oldkey == (C1, C2):
                    continue
                newkey = oldkey
                if A == C2:
                    newkey = (C1, B)
                    if C1 > B:
                        newkey = (B, C1)
                remove_elem(sigma_to_C1C2, C1C2_to_sigma, oldkey)
                if not (newkey, B) in updatePairs:
                    updatePairs.add((newkey, B))
                if (A == C2):
                    Cneig[B].remove(C2)
                    Cneig[B].add(C1)


        # update the sigma mappings
        for (newpair, C) in updatePairs:
            newsigma = delta_sigma_C3C(n, Ccard[C1], Ccard[C2], Ccard[C], D, CP_t[C1], CP_t[C2], CP_t[C], sigmaC1C2)
            add_elem(sigma_to_C1C2, C1C2_to_sigma, newpair, newsigma)

        # finally update the relevant values of the sets
        CP_t[C1] = P_t_C3
        del CP_t[C2]
        neigC3 = Cneig[C1].union(Cneig[C2])
        neigC3.remove(C1)
        neigC3.remove(C2)
        Cneig[C1] = neigC3
        del Cneig[C2]
        Ccard[C1] = cardC3
        del Ccard[C2]


    dendomatrix = []
    oldtonewset = dict()
    currnew = n
    for (a, b, sigma) in merge_order:
        newa = a
        newb = b
        if a in oldtonewset:
            newa = oldtonewset[a]
        if b in oldtonewset:
            newb = oldtonewset[b]

        if newa < newb:
            dendomatrix.append([newa, newb, currnew, sigma])
        else:
            dendomatrix.append([newb, newa, currnew, sigma])

        oldtonewset[a] = currnew
        currnew += 1

    fig = plt.figure(figsize=(25, 10))
    fancy_dendrogram(
        dendomatrix,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=0.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=40,
        max_d=170,
    )
    #dn = dendrogram(dendomatrix)
    plt.show()

    return


if __name__ == '__main__':
    main()
