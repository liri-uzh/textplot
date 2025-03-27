#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:
    python -m examples.build_graph examples/corpora/war-and-peace.txt

"""

import sys
from textplot.helpers import build_graph
import networkx as nx

if __name__ == "__main__":

    infile = sys.argv[1]
    
    g = build_graph(str(infile))
    
    print(nx.degree_centrality(g.graph))