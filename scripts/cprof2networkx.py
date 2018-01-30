#!/usr/bin/env python

import sys
import json
import subprocess
import networkx as nx
import numpy as np

import pycprof

g = nx.DiGraph()


def estTimeMemcpyH2d(txSize):
    x = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
         1.05E+06, 2.10E+06, 4.19E+06, 8.39E+06, 1.68E+07, 3.36E+07, 6.71E+07, 1.34E+08, 2.68E+08, 5.37E+08]
    bw = [4.3553, 8.68704, 17.2806, 34.0123, 71.7322, 141.901, 271.758, 509.223, 920.309, 1501.97, 2191.91, 2886.7,
          3838.48, 4476.04, 4894.52, 4976.09, 5508.2, 5782.67, 5878.86, 5975.69, 5976.23, 5773.07, 6017.5, 5849.2, 5856.51]
    return np.interp(txSize, x, bw)


def get_value_node_id(i):
    return "val" + str(i)


def get_api_node_id(i):
    return "api" + str(i)


def get_node_id(n):
    if type(n) == pycprof.Value:
        return get_value_node_id(n.id_)
    elif type(n) == pycprof.API:
        return get_api_node_id(n.id_)
    else:
        print type(n)
        assert False


def value_handler(val):
    if type(val) != pycprof.Value:
        return
    valNodeId = get_node_id(val)
    g.add_node(valNodeId, node=val)

# Add a node for each api. Connect values to apis and vis versa


def api_handler(api):
    if type(api) != pycprof.API:
        return

    apiNodeId = get_node_id(api)

    if "cudaLaunch" in api.functionName:
        g.add_node(apiNodeId, node=api)
        for i in api.inputs:
            srcNodeId = get_value_node_id(i)
            weight = g.node[srcNodeId]['node'].size
            # api node deps on src node
            g.add_edge(apiNodeId, srcNodeId, Weight=weight)

        for o in api.outputs:
            dstNodeId = get_value_node_id(o)
            weight = g.node[dstNodeId]['node'].size
            g.add_edge(dstNodeId, apiNodeId, Weight=weight)
    elif "cudaMemcpy" == api.functionName:
        assert len(api.inputs) == 1
        assert len(api.outputs) == 1
        srcNodeId = get_value_node_id(api.inputs[0])
        dstNodeId = get_value_node_id(api.outputs[0])
        g.add_edge(dstNodeId, srcNodeId, Weight=api.dstCount)


def dep_handler(dep):
    return
    if type(dep) != pycprof.Dependence:
        return
    src = dep.src
    dst = dep.dst
    srcValId = get_value_node_id(src)
    dstValId = get_value_node_id(dst)
    # g.add_edge(srcNode, dstNode, directed=True)


pycprof.run_handler(value_handler, path=sys.argv[1])
pycprof.run_handler(api_handler, path=sys.argv[1])
pycprof.run_handler(dep_handler, path=sys.argv[1])

# print "cycles"
# cycles = nx.find_cycle(g)
# print cycles
# print "done"


print "longest path:"
longestPath = nx.dag_longest_path(g)
baseCost = 0.0
optCost = 0.0
for i in range(len(longestPath) - 1):
    srcIdx = i
    dstIdx = i + 1
    srcNodeId = longestPath[srcIdx]
    dstNodeId = longestPath[dstIdx]
    srcNode = g.node[srcNodeId]['node']
    dstNode = g.node[dstNodeId]['node']
    print srcNodeId, srcNode, dstNodeId, dstNode
    if type(srcNode) == pycprof.Value:
        if type(dstNode) == pycprof.Value:
            baseCost += srcNode.size
        elif type(dstNode) == pycprof.API:
            baseCost += srcNode.size
            optCost += srcNode.size
    elif type(srcNode) == pycprof.API:
        if type(dstNode) == pycprof.Value:
            baseCost += dstNode.size
            optCost += dstNode.size

print len(longestPath), baseCost, optCost, optCost / baseCost
print "done"

print "writing graph...",
nx.write_graphml(g, "cprof.graphml")
print "done!"
