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

def main():
    k = 2           # number of communities
    cardC = [5, 5]  # community sizes
    dinC = [8, 9]   # internal degree of each community
    N = 0           # total number of vertices
    Qe = 0.2        # wanted modularity - determines the value of beta
    return

if __name__ == '__main__':
    main()