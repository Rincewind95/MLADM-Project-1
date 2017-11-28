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
from collections import defaultdict
from collections import namedtuple
import matplotlib.pyplot as plt
import datetime
import sys
import numpy as np

# special struct used for the sparse vector
SparseVec = namedtuple('SparseVec', ['indices', 'value'])

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


def fancy_plot(n, merge_order, reverse_mapping):
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


def load_graph(location):
    print_with_timestep('Loading G...')
    data = np.genfromtxt(location, skip_header=1, dtype=int)
    G = dict()
    points = SortedSet()
    for c in data:
        x = c[0]
        y = c[1]
        points.add(x)
        points.add(y)
    n = len(points)
    mapping = dict()
    reverse_mapping = dict()
    for i in range(n):
        mapping[points[i]] = i
        reverse_mapping[i] = points[i]

    for c in data:
        x = mapping[c[0]]
        y = mapping[c[1]]
        if not x in G:
            G[x] = SortedSet()
        if not y in G:
            G[y] = SortedSet()
        G[x].add(y)
        G[y].add(x)
    for x in range(n):
        G[x].add(x)
    print_with_timestep('Finished loading G...')
    return (G, n, reverse_mapping)


def get_sparse_identity(n):
    # artificially modify G to have 'self-loops'
    indices = []
    indexptr = [0]
    data = []
    for curr in range(n):
        indices.append(curr)
        data.append(1.0)
        indexptr.append(len(indices))

    S = csr_matrix((data, indices, indexptr), dtype=float)
    return S


def exp_squaring(P, t, n):
    R = get_sparse_identity(n) # np.identity(node_cnt, dtype='float')
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


def get_P_t(Gself, t, n):
    print_with_timestep('Calculating P...')
    indices = []
    indexptr = [0]
    data = []
    for curr in range(n):
        neig_cnt = len(Gself[curr])
        for neig in Gself[curr]:
            indices.append(neig)
            data.append(1.0/neig_cnt)
        indexptr.append(len(indices))

    P = csr_matrix((data, indices, indexptr), dtype=float)
    print_with_timestep('Sqaring P...')
    P = exp_squaring(P, t, n)
    print_with_timestep('Finished sqaring P...')
    return P


def sorted_union(a, b):
    idxa = 0
    idxb = 0
    l = []
    lena = len(a)
    lenb = len(b)
    while idxa < lena and idxb < lenb:
        if a[idxa] < b[idxb]:
            l.append(a[idxa])
            idxa += 1
        elif a[idxa] > b[idxb]:
            l.append(b[idxb])
            idxb += 1
        else:
            l.append(a[idxa])
            idxa += 1
            idxb += 1
    while idxa < lena:
        l.append(a[idxa])
        idxa += 1
    while idxb < lenb:
        l.append(b[idxb])
        idxb += 1
    return l


def get_r2_C1C2(DP_t_C1, DP_t_C2):
    res = 0
    indices = sorted_union(DP_t_C1.indices, DP_t_C2.indices)
    for index in indices:
        elem = DP_t_C1.value[index] - DP_t_C2.value[index]
        res += elem*elem
    return res


def delta_sigma_C1C2(n, cardC1, cardC2, DP_t_C1, DP_t_C2):
    r2_C1C2 = get_r2_C1C2(DP_t_C1, DP_t_C2)
    return ((cardC1 * cardC2) / ((cardC1 + cardC2)*n))*r2_C1C2


def delta_sigma_C3C(Ccard, C1, C2, C, oldkeyMap, deltaC1C2, CDP_t, DP_t_C3, n):
    C1C = get_min_pair(C1, C)
    C2C = get_min_pair(C2, C)
    if C1C in oldkeyMap and C2C in oldkeyMap:
        C1C = (Ccard[C1] + Ccard[C]) * oldkeyMap[C1C]
        C2C = (Ccard[C2] + Ccard[C]) * oldkeyMap[C2C]
        C1C2 = Ccard[C] * deltaC1C2
        return (C1C + C2C - C1C2) / (Ccard[C1] + Ccard[C2] + Ccard[C])
    else:
        return delta_sigma_C1C2(n, Ccard[C1] + Ccard[C2], Ccard[C], DP_t_C3, CDP_t[C])

def remove_elem(sigma_to_C1C2, C1C2_to_sigma, oldkeyMap, oldkey):
    sigmaC1C2 = C1C2_to_sigma[oldkey]
    oldkeyMap[oldkey] = sigmaC1C2
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


def get_min_pair(C1, C2):
    if C1 < C2:
        return (C1, C2)
    else:
        return (C2, C1)


#infilename = 'com-amazon.ungraph'
infilename = 'example'
#infilename = '18-18-55_gen_graph'
outfilename = '%s_output' % '{:%H-%M-%S}'.format(datetime.datetime.now())
calcfilename = '%s_calc_%s' % ('{:%H-%M-%S}'.format(datetime.datetime.now()),infilename)
outdir = 'out/'
indir = 'testing/examples/'
#indir = 'testing/amazon/'
txt = '.txt'
infilename = indir + infilename + txt
outfilename = outdir + outfilename + txt
calcfilename = outdir + calcfilename + txt
outfile = open(outfilename, 'w')
calcfile = open(calcfilename, 'w')

def increment_visual_counter(visual_counter):
    visual_counter += 1
    if visual_counter % 1000 == 0:
        print_with_timestep("%sk done..." % (visual_counter / 1000))
    return visual_counter


def main():
    t = 3                            # choice of t
    print_no_newline('!IMPORTANT! -> t = %s\n\n' % t)

    # retrieve the graph with self-edges, total number of vertices and mapping to original indices
    (G, n, reverse_mapping) = load_graph(infilename)

    P_t = get_P_t(G, t, n)           # the Pt matrix from the question set and
    Deg = dict()                     # degree of each vertex
    Ccard = dict()                   # the mapping of community indices to their respective cardinality
    CDP_t = dict()                   # the mapping of community indices to their respective D^-(1/2)*P_t_C.
    Cneig = dict()                   # the mapping of community indices to their "neighbour" communities (ones who share a direct edge to them)
    C1C2_to_sigma = dict()           # the mapping of two adjacent community indices to their respective sigma (assumed to be unique enough!)
    unsorted_sigma_to_C1C2 = dict()  # a mapping of sigmas (updated with new values regularly)
    merge_order = []                 # a list of tuples in set merge order ((C1,C2) means C1 and C2 were merged to C1)

    print_with_timestep('Calculating D and smaller things...')

    for v in range(n):
        Ccard[v] = 1.0
        Deg[v] = len(G[v])-1
        Cneig[v] = set()
        for elem in G[v]:
            Cneig[v].add(elem)
        Cneig[v].remove(v) # remove the redundant "self" element from the set

    print_with_timestep('Calculating D and DP_t...')
    D = dict()
    for v in range(n):
        D[v] = 1.0 / sqrt(Deg[v] + 1)


    visual_counter = 0
    for v in range(n):
        visual_counter = increment_visual_counter(visual_counter)
        pt = P_t[v, :]
        indexes = pt.indices
        data = pt.data
        CDP_t[v] = SparseVec(indices=[],
                             value=defaultdict(lambda: 0.0))
        for (pos, index) in enumerate(indexes):
            CDP_t[v].indices.append(index)
            CDP_t[v].value[index] = data[pos] * D[index]
        CDP_t[v].indices.sort()

    print_with_timestep('Finished calculating D and DP_t...')

    print_with_timestep('Explicitly delete G, P_t and Deg...')
    # explicitly get rid of G to cut down on memory
    for v in range(n):
        G[v] = None
    del G
    del P_t
    del Deg
    print_with_timestep('Finished deleting G, P_t and Deg...')

    print_with_timestep('Setting up the structures...')
    visual_counter = 0
    for C1 in range(n):
        for C2 in Cneig[C1]:
            visual_counter = increment_visual_counter(visual_counter)

            if not (C1, C2) in C1C2_to_sigma and C1 < C2:
                sigmaC1C2 = delta_sigma_C1C2(n, Ccard[C1], Ccard[C2], CDP_t[C1], CDP_t[C2])
                C1C2_to_sigma[(C1, C2)] = sigmaC1C2
                if sigmaC1C2 not in unsorted_sigma_to_C1C2:
                    unsorted_sigma_to_C1C2[sigmaC1C2] = SortedSet()
                unsorted_sigma_to_C1C2[sigmaC1C2].add((C1, C2))

    sigma_to_C1C2 = SortedDict(unsorted_sigma_to_C1C2) # a constantly sorted mapping of sigmas (updated with new values regularly)

    visual_counter = 0
    # iterate through all merges
    print_with_timestep('About to start the algo...')
    for _ in range(n-1):
        visual_counter = increment_visual_counter(visual_counter)

        # select the minimum element in the sorted set, record and remove it
        (C1, C2) = sigma_to_C1C2.viewvalues()[0][0]
        sigmaC1C2 = C1C2_to_sigma[(C1,C2)]
        merge_order.append((C1, C2, sigmaC1C2))
        del C1C2_to_sigma[(C1,C2)]
        del sigma_to_C1C2[sigmaC1C2][0]
        if len(sigma_to_C1C2[sigmaC1C2]) <= 0:
            del sigma_to_C1C2[sigmaC1C2]

        # calculate the values for the new community
        DP_t_C3 = SparseVec(indices=sorted_union(CDP_t[C1].indices, CDP_t[C2].indices),
                            value=defaultdict(lambda: 0.0))
        c1card = Ccard[C1]
        c2card = Ccard[C2]
        cardC3 = Ccard[C1] + Ccard[C2]
        for index in DP_t_C3.indices:
            DP_t_C3.value[index] = (c1card*CDP_t[C1].value[index] + c2card*CDP_t[C2].value[index])/cardC3

        # determine which tuples need updating and resolve maintenance things
        updatePoints = [C1, C2]
        oldkeyMap = dict() # saved mapping of old keys to sigmas
        updatePairs = set()
        for A in updatePoints:
            neighbours = Cneig[A]
            for B in neighbours:
                oldkey = get_min_pair(A, B)
                if oldkey == (C1, C2):
                    continue
                newkey = oldkey
                if A == C2:
                    newkey = get_min_pair(C1, B)
                remove_elem(sigma_to_C1C2, C1C2_to_sigma, oldkeyMap, oldkey)
                if not (newkey, B) in updatePairs:
                    updatePairs.add((newkey, B))
                if (A == C2):
                    Cneig[B].remove(C2)
                    Cneig[B].add(C1)

        # update the sigma mappings
        for (newpair, C) in updatePairs:
            newsigma = delta_sigma_C3C(Ccard, C1, C2, C, oldkeyMap, sigmaC1C2, CDP_t, DP_t_C3, n)
            add_elem(sigma_to_C1C2, C1C2_to_sigma, newpair, newsigma)

        # finally update the relevant values of the sets
        CDP_t[C1] = DP_t_C3
        del CDP_t[C2]
        neigC3 = Cneig[C1].union(Cneig[C2])
        neigC3.remove(C1)
        neigC3.remove(C2)
        Cneig[C1] = neigC3
        del Cneig[C2]
        Ccard[C1] = cardC3
        del Ccard[C2]
    print_with_timestep('Algo completed...')

    print_no_newline('\n')
    print_with_timestep('Merging order:')
    for (idx, (a, b, sigma)) in enumerate(merge_order):
        print_no_newline('%s: [%s, %s, %s]\n' % (idx, a, b, sigma))

    # compute the optimal partitioning according to metric eta
    sigma_prev = 0
    max_eta = 0
    max_iter = 0
    for (idx, (a, b, sigma)) in enumerate(merge_order):
        if idx == 0:
            sigma_prev = sigma
        else:
            eta = sigma/sigma_prev
            if eta > max_eta:
                max_eta = eta
                max_iter = idx
            sigma_prev = sigma

    sets = SortedDict()
    for i in range(n):
        sets[i] = SortedSet()
        sets[i].add(reverse_mapping[i])

    for i in range(max_iter):
        (a, b, sigma) = merge_order[i]
        sets[a] = sets[a].union(sets[b])
        del sets[b]

    print_no_newline('\n')
    print_with_timestep('Optimal solution (eta = %s, max_iter = %s):' % (max_eta, max_iter))
    for idx in sets:
        for i in range(len(sets[idx])):
            print_calcfile(str(sets[idx][i]))
            if i < len(sets[idx]) - 1:
                print_calcfile(' ')
        print_calcfile('\n')

    #fancy_plot(n, merge_order, reverse_mapping)

    outfile.close()
    calcfile.close()
    return

def print_no_newline(string):
    sys.stdout.write(string)
    outfile.write(string)
    sys.stdout.flush()
    outfile.flush()

def print_calcfile(string):
    sys.stdout.write(string)
    outfile.write(string)
    calcfile.write(string)
    sys.stdout.flush()
    outfile.flush()
    calcfile.flush()

def print_with_timestep(string):
    s = '%s: %s\n' % ('{:%H:%M:%S.%f}'.format(datetime.datetime.now()), string)
    sys.stdout.write(s)
    outfile.write(s)
    sys.stdout.flush()
    outfile.flush()

if __name__ == '__main__':
    main()
