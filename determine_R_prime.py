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

def parsefile(infile):
    data = np.genfromtxt(infile, skip_header=1, dtype=int)
    N = 0
    G = dict()
    points = set()
    for (idx, line) in enumerate(data):
        G[idx] = set()
        for elem in line:
            G[idx].add(elem)
        N += len(G[idx])
    return (G, N)

def main():
    infilename_base = '18-18-55_gen_calc'
    infilename_calc = '19-55-16_calc_18-18-55_gen_graph'
    base_folder = 'out/'
    outfilename = 'out/Rprime_%s_-_%s.txt' % (infilename_base, infilename_calc)
    infilename_base_full = base_folder + infilename_base + '.txt'
    infilename_calc_full = base_folder + infilename_calc + '.txt'

    (G_base, N1) = parsefile(infilename_base_full)
    (G_calc, N2) = parsefile(infilename_calc_full)

    outfile = open(outfilename, "w")

    if N1 != N2:
        outfile.write('ERROR -> N1 != N2')
        return

    N = N1
    SumNormC1i2 = 0
    SumNormC2j2 = 0
    for C1i in G_base:
        cnt = len(G_base[C1i])
        SumNormC1i2 += cnt*cnt
    for C2j in G_calc:
        cnt = len(G_calc[C2j])
        SumNormC2j2 += cnt*cnt

    SumC1i_inter_C2j2 = 0
    for C1i in G_base:
        for C2j in G_calc:
            Cinter = G_base[C1i].intersection(G_calc[C2j])
            cnt = len(Cinter)
            SumC1i_inter_C2j2 += cnt*cnt

    Rprime = (N*N*SumC1i_inter_C2j2 - SumNormC1i2*SumNormC2j2)/(0.5*N*N*(SumNormC1i2 + SumNormC2j2) - SumNormC1i2*SumNormC2j2)

    outfile.write('R\' = %s' % Rprime)
    return Rprime

if __name__ == '__main__':
    main()