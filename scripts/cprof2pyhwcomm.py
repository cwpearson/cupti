#!/usr/bin/env python

import sys
import networkx as nx
import matplotlib as plt

from pycprof.profile import Profile
import pyhwcomm.machines.minsky
import pyhwcomm

# from pyhwcomm.machines.minsky import Minsky

ALLOCS = {}
VALUES = {}
APIS = {}

minsky = pyhwcomm.machines.minsky.Minsky()


def loc_to_minsky_device(loc):
    if loc.type == "cuda":
        dst = minsky.cuda_gpu()[loc.id_]
    elif loc.type == "host":
        if loc.id_ == -1:
            loc.id_ = 0
        dst = minsky.cpu()[loc.id_]
    else:
        assert False
    return dst


p = Profile(sys.argv[1])
print len(p.graph.nodes)

# nx.draw(p.graph)
# plt.show()
