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
    results = []
    with open(infile) as filein:
        for line in filein:
            results.append(line.strip().split(' '))

    data = []
    for (idx,row) in enumerate(results):
        data.append([])
        for i in row:
            data[idx].append(int(i))

    N = 0
    G = dict()
    for (idx, line) in enumerate(data):
        G[idx] = set()
        for elem in line:
            G[idx].add(elem)
        N += len(G[idx])
    return (G, N)

def main():
    #infilename_base = 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc'
    #infilename_calc = 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t1_calc'

    base_folder = 'in/GRAPH3extra/'
    calc_folder = 'out/GRAPH3extra/'
    outfilename = 'out/RprimeGRAPH3extra.txt'
    outfile = open(outfilename, "w")

    # for (infilename_base, infilename_calc) in \
    # [ \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t1_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t2_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t3_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t4_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t5_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t6_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t7_calc' ), \
    #     ( 'nreq600_k2_alpha2_qe0.4_(n600)_gen_calc', 'nreq600_k2_alpha2_qe0.4_(n600)_gen_graph_t8_calc' )
    # ]:

    #GRAPH1
    # for infilename_root in \
    # [ \
    #         'nreq600_k2_alpha2_qe0.4_(n600)', \
    #         'nreq600_k3_alpha2_qe0.4_(n600)', \
    #         'nreq600_k5_alpha2_qe0.4_(n600)', \
    #         'nreq600_k6_alpha2_qe0.4_(n600)', \
    #         'nreq600_k12_alpha2_qe0.4_(n600)', \
    #         'nreq600_k24_alpha2_qe0.4_(n600)', \
    #         'nreq600_k36_alpha2_qe0.4_(n576)' \
    # ]:

    # GRAPH2
    # for infilename_root in \
    # [ \#
    #     'nreq300_k10_alpha2_qe0.4_(n300)', \
    #     'nreq600_k10_alpha2_qe0.4_(n600)', \
    #     'nreq900_k10_alpha2_qe0.4_(n900)', \
    #     'nreq1200_k10_alpha2_qe0.4_(n1200)' \
    # ]:

    # GRAPH3
    # for infilename_root in \
    # [ \
    #     'nreq600_k5_alpha2_qe0.1_(n6# 00)', \
    #     'nreq600_k5_alpha2_qe0.2_(n600)', \
    #     'nreq600_k5_alpha2_qe0.3_(n600)', \
    #     'nreq600_k5_alpha2_qe0.4_(n600)', \
    #     'nreq600_k5_alpha2_qe0.5_(n600)', \
    #     'nreq600_k5_alpha2_qe0.6_(n600)', \
    #     'nreq600_k5_alpha2_qe0.7_(n600)' \
    # ]:

    # GRAPH3extra
    for infilename_root in \
    [ \
        'nreq600_k5_alpha2_qe0.32_(n600)', \
        'nreq600_k5_alpha2_qe0.34_(n600)', \
        'nreq600_k5_alpha2_qe0.36_(n600)', \
        'nreq600_k5_alpha2_qe0.38_(n600)' \
    ]:

    # GRAPH4
    # for infilename_root in \
    # [ \
    #   'nreq600_k5_alpha1_qe0.4_(n600)', \
    #   'nreq600_k5_alpha2_qe0.4_(n600)', \
    #   'nreq600_k5_alpha3_qe0.4_(n600)', \
    #   'nreq600_k5_alpha4_qe0.4_(n600)', \
    #   'nreq600_k5_alpha5_qe0.4_(n600)', \
    #   'nreq600_k5_alpha6_qe0.4_(n600)', \
    #   'nreq600_k5_alpha7_qe0.4_(n600)', \
    # ]:

        outfile.write('t\tR\' %s\n' % infilename_root)
        for t in [ 1, 2, 3, 4, 5, 6, 7, 8 ]:

            infilename_base_full = base_folder + infilename_root + '_gen_calc.txt'
            infilename_calc_full = calc_folder + infilename_root + '_gen_graph_t' + str(t) + '_calc.txt'

            #infilename_base_full = base_folder + infilename_base + '.txt'
            #infilename_calc_full = calc_folder + infilename_calc + '.txt'

            (G_base, N1) = parsefile(infilename_base_full)
            (G_calc, N2) = parsefile(infilename_calc_full)

            if N1 != N2:
                outfile.write('%s\tERROR -> N1 != N2\n' % t )
            else:
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

                print 'R\' = %s' % Rprime
                outfile.write('%s\t%s\n' % (t, Rprime) )

        outfile.write('\n')

    outfile.close()
    return

if __name__ == '__main__':
    main()