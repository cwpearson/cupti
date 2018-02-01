#!/usr/bin/env python

import sys
import json
import subprocess
import networkx as nx
import numpy as np

import pycprof

g = nx.DiGraph()
ALLOCS = {}
APIS = {}


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


def alloc_handler(alloc):
    ALLOCS[alloc.id_] = alloc


def value_handler(val):
    valNodeId = get_node_id(val)
    g.add_node(valNodeId, node=val)


def api_handler(api):
    APIS[api.id_] = api
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


def api_handler2(api):
    APIS[api.id_] = api


def dep_handler(dep):
    src = dep.src
    dst = dep.dst
    cause = dep.api_cause
    srcValId = get_value_node_id(src)
    dstValId = get_value_node_id(dst)
    srcNode = g.node[srcValId]['node']
    dstNode = g.node[dstValId]['node']
    srcAllocId = srcNode.allocation_id
    dstAllocId = dstNode.allocation_id
    srcAlloc = ALLOCS[srcAllocId]
    dstAlloc = ALLOCS[dstAllocId]
    srcLoc = srcAlloc.loc
    dstLoc = dstAlloc.loc
    if srcLoc != dstLoc:
        # print srcLoc, srcNode.size, dstLoc, dstNode.size, APIS[apiCauseId].functionName
        cost = dstNode.size
    else:
        cost = 0

    g.add_edge(srcValId, dstValId, Weight=dstNode.size,
               Cause=cause, Cost=cost)


pass1 = {pycprof.Allocation: alloc_handler,
         pycprof.Value: value_handler,
         pycprof.API: api_handler2}
pass2 = {pycprof.Dependence: dep_handler}
pycprof.run_pass(pass1, path=sys.argv[1])
pycprof.run_pass(pass2, path=sys.argv[1])

# print "cycles"
# cycles = nx.find_cycle(g)
# print cycles
# print "done"


sources = [n for n, d in g.in_degree() if d == 0]
sinks = [n for n, d in g.out_degree() if d == 0]
print len(sources), "sources"
print len(sinks), "sinks"


def cost(g, path):
    weight = 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        srcNodeId = path[i]
        dstNodeId = path[i + 1]
        weight += g[srcNodeId][dstNodeId]['Weight']

        srcNode = g.node[srcNodeId]['node']
        dstNode = g.node[dstNodeId]['node']
        apiCauseId = g[srcNodeId][dstNodeId]["Cause"]

        if isinstance(srcNode, pycprof.Value):
            if isinstance(dstNode, pycprof.Value):
                srcAllocId = srcNode.allocation_id
                dstAllocId = dstNode.allocation_id
                srcAlloc = ALLOCS[srcAllocId]
                dstAlloc = ALLOCS[dstAllocId]
                srcLoc = srcAlloc.loc
                dstLoc = dstAlloc.loc
                if srcLoc != dstLoc:
                    # print srcLoc, srcNode.size, dstLoc, dstNode.size, APIS[apiCauseId].functionName
                    cost += dstNode.size

    # print weight, cost
    return cost


# maxCost = 0
# for sourceNode in sources:
#     for sinkNode in sinks:
#         simple_paths = nx.all_simple_paths(g, sourceNode, sinkNode)

#         for path in simple_paths:
#             print sourceNode, sinkNode
#             c = cost(g, path)
#             if c > maxCost:
#                 maxCost = c
#                 print c, len(path), sourceNode, sinkNode

# sys.exit(-1)

print "longest path:"
longestPath = nx.dag_longest_path(g, weight="Cost")
baseCost = 0.0
for i in range(len(longestPath) - 1):
    srcIdx = i
    dstIdx = i + 1
    srcNodeId = longestPath[srcIdx]
    dstNodeId = longestPath[dstIdx]
    srcNode = g.node[srcNodeId]['node']
    dstNode = g.node[dstNodeId]['node']
    apiCauseId = g[srcNodeId][dstNodeId]["Cause"]
    print srcNodeId + " -> " + dstNodeId, g[srcNodeId][dstNodeId]['Cost']
    if isinstance(srcNode, pycprof.Value):
        if isinstance(dstNode, pycprof.Value):
            srcAllocId = srcNode.allocation_id
            dstAllocId = dstNode.allocation_id
            srcAlloc = ALLOCS[srcAllocId]
            dstAlloc = ALLOCS[dstAllocId]
            srcLoc = srcAlloc.loc
            dstLoc = dstAlloc.loc
            if srcLoc != dstLoc:
                print srcLoc, srcNode.size, dstLoc, dstNode.size, APIS[apiCauseId].functionName
                baseCost += dstNode.size

print len(longestPath), baseCost
print "done"

print "writing graph...",
nx.write_graphml(g, "cprof.graphml")
print "done!"
