from networkx import Graph
import os

import networkx as nx
from networkx import Graph
from scipy import *
from sortedcontainers import SortedSet


def get_beta(Qe, dinC, k):
    top = 0
    bot = 0
    for Ci in range(k):
        tmp = dinC[Ci]
        top += tmp * tmp
        bot += tmp
    bot = bot * bot
    return (bot*(1-Qe) - top)/(bot*Qe+top)


def generate( Nreq, k, alpha, Qe ):
    cardC = []
    dinC = []
    doutC = []

    for i in range(k):
        cardC.append(int(Nreq / k))
        dinC.append(cardC[i] * int(alpha * log(cardC[i])) / 2)

    idx_lbo = [0]  # the lower bound for indices in each cluster
    for i in range(1, k):
        idx_lbo.append(idx_lbo[i - 1] + cardC[i])

    N = 0
    for i in range(k):
        N += cardC[i]

    beta = get_beta(Qe, dinC, k)

    for i in range(k):
        doutC.append(int(dinC[i] * beta))

    dir = 'out/'
    txt = '.txt'
    outfilebase = 'nreq%s_k%s_alpha%s_qe%s_(n%s)_gen' % ( Nreq, k, alpha, Qe, N )
    outfilepath = dir + outfilebase + '_graph' + txt

    print "...generating %s" % outfilebase

    if os.path.isfile( outfilepath ):
        print "--- SKIPPED: file already exists."
        return

    all_edges = []
    if beta > 0:
        # determine all the internal edges
        lef = 0
        rig = 0
        graph_exists = True
        for Ci in range(k):
            lef = rig
            rig += cardC[Ci]

            all_curr_edges = []
            for i in range(lef, rig - 1):
                for j in range(i + 1, rig):
                    all_curr_edges.append((i, j))

            curr_selected_edges = []
            all_curr_edge_cnt = len(all_curr_edges)
            if dinC[Ci] < all_curr_edge_cnt:
                curr_selected_indices = np.random.choice(range(all_curr_edge_cnt), dinC[Ci], replace=False)
                for i in curr_selected_indices:
                    curr_selected_edges.append(all_curr_edges[i])
            elif dinC[Ci] == all_curr_edge_cnt:
                curr_selected_edges = all_curr_edges
            else:
                graph_exists = False
                print "### FAILED: dinC[Ci] > all_curr_edge_cnt"
                break

            for i in range(len(curr_selected_edges)):
                all_edges.append(curr_selected_edges[i])

        if graph_exists:
            # now on to the external ones
            remaining_doutC = dict()
            for i in range(k):
                remaining_doutC[i] = doutC[i]
            inter_connections = SortedSet()
            for C1 in range(k - 1):
                for C2 in range(C1 + 1, k):
                    inter_connections.add((C1, C2))

            # determine the "bundles" of edges between two clusters
            external_edge_bundle = dict()
            while remaining_doutC and len(inter_connections) > 0:
                curr_cnt = len(inter_connections)
                choice = np.random.choice(range(curr_cnt))
                (C1, C2) = inter_connections[choice]

                max_alloc = minimum(remaining_doutC[C1], remaining_doutC[C2])

                if max_alloc == 0:
                    graph_exists = False
                    print "### FAILED: cluster has no outgoing edges"
                    break

                alloc = np.random.choice(range(1, max_alloc + 1))

                if (C1, C2) not in external_edge_bundle:
                    external_edge_bundle[(C1, C2)] = 0
                external_edge_bundle[(C1, C2)] += alloc

                options = [C1, C2]
                for C in options:
                    remaining_doutC[C] -= alloc
                    if remaining_doutC[C] == 0:
                        del remaining_doutC[C]
                        for otherC in range(k):
                            pair = (minimum(C, otherC), maximum(C, otherC))
                            if pair in inter_connections:
                                inter_connections.remove(pair)
            if graph_exists:
                for (C1, C2) in external_edge_bundle:
                    all_curr_edges = []
                    for i in range(idx_lbo[C1], idx_lbo[C1] + cardC[C1]):
                        for j in range(idx_lbo[C2], idx_lbo[C2] + cardC[C2]):
                            all_curr_edges.append((i, j))

                    allowance = external_edge_bundle[(C1, C2)]
                    max_edges = len(all_curr_edges)
                    if allowance > max_edges:
                        print " ### FAILED: allowance > max_edges"
                        graph_exists = False
                        break
                    curr_selected_indices = np.random.choice(range(max_edges), allowance, replace=False)

                    curr_selected_edges = []
                    for i in curr_selected_indices:
                        curr_selected_edges.append(all_curr_edges[i])

                    for i in range(len(curr_selected_edges)):
                        all_edges.append(curr_selected_edges[i])

        if graph_exists:
            G = Graph()
            for (a,b) in all_edges:
                G.add_edge(a, b)
                G.add_edge(b, a)
            if not nx.is_connected(G):
                print "### FAILED: graph not connected"
                graph_exists = False

        # G = Graph()
        # for (v1, v2) in all_edges:
        #    G.add_edge(v1, v2)
        # nx.draw(G)

        if graph_exists:
            outfile = open(outfilepath, 'w')
            outfile.write("# k = %s\n" % k)
            outfile.write("# N = %s\n" % N)
            outfile.write("# alpha = %s\n" % alpha)
            outfile.write("# beta = %s\n" % beta)
            # outfile.write("# gamma = %s\n" % gamma)
            outfile.write("# Qe = %s\n" % Qe)
            outfile.write("#\tFromNodeId\tToNodeId\n")
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
    else:
        print "### FAILED: beta < 0"

    return

def main():
    # mysmall example#
    # k = 4                 # number of communities
    # cardC = [5, 5, 5, 5]  # community sizes
    # dinC = [8, 7, 6, 7]   # internal degree of each community
    # doutC = []      # external degree of each community
    # N = 0           # total number of vertices
    # Qe = 0.2        # wanted modularity - determines the value of beta




    # if the graph is homogenious
    # Nreq = 300                         # n300, n1000, n3000, n10000
    # gamma = 0.2                        # g02,  g04,  g06,  g08
    # k = int(round(pow(Nreq, gamma)))   # k3,   k10,  k30,  k96   <- n300
                                         # k4,   k16,  k63,  k251  <- n1000
                                         # k5,   k25,  k122, k605  <- n3000
                                         # k6,   k40,  k251, k1584 <- n10000

    # alpha = 2                          # a2, a4, a6
    # Qe = 0.2                           # q02, q04, q06,     q08

    # if the graph is homogenious
    # for Nreq in [ 300, 1000, 3000, 10000 ]:
    #     for gamma in [ 0.2, 0.4, 0.6, 0.8 ]:
    #         for alpha in [ 2, 4, 6 ]:
    #             for Qe in [ 0.2, 0.4, 0.6, 0.8 ]:
    #                 generate(Nreq, gamma, alpha, Qe)

    # GRAPH1
    # for Nreq in [ 600 ]:
    #     for k in [ 2, 3, 5, 6, 12, 24, 36 ]:
    #         for alpha in [ 2 ]:
    #             for Qe in [ 0.4 ]:
    #                 generate(Nreq, k, alpha, Qe)

    # GRAPH2
    # for Nreq in [ 300, 600, 900, 1200 ]:
    #     for k in [ 10 ]:
    #         for alpha in [ 2 ]:
    #             for Qe in [ 0.4 ]:
    #                 generate(Nreq, k, alpha, Qe)

    # GRAPH3
    # for Nreq in [ 600 ]:
    #     for k in [ 5 ]:
    #         for alpha in [ 2 ]:
    #             for Qe in [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ]:
    #                 generate(Nreq, k, alpha, Qe)

    # GRAPH3 extra
    for Nreq in [ 600 ]:
        for k in [ 5 ]:
            for alpha in [ 2 ]:
                for Qe in [ 0.32, 0.34, 0.36, 0.38 ]:
                    generate(Nreq, k, alpha, Qe)

    # GRAPH4
    # for Nreq in [ 600 ]:
    #     for k in [ 5 ]:
    #         for alpha in [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]:
    #    #          for Qe in [ 0.4 ]:
    #                 generate(Nreq, k, alpha, Qe)

    return

if __name__ == '__main__':
    main()