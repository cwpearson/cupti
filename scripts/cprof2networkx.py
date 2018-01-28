#!/usr/bin/env python

import sys
import json
import subprocess
import networkx as nx

import pycprof

g = nx.DiGraph()

NODE_WEIGHT = "Weight"
EDGE_WEIGHT = "Weight"


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
    g.add_node(valNodeId)
    g.node[valNodeId][NODE_WEIGHT] = float(val.size)

# Add a node for each api. Connect values to apis and vis versa


def api_handler(api):
    if type(api) != pycprof.API:
        return
    apiNodeId = get_node_id(api)
    g.add_node(apiNodeId)
    for i in api.inputs:
        srcNodeId = get_value_node_id(i)
        weight = g.node[srcNodeId][NODE_WEIGHT]
        # api node deps on src node
        g.add_edge(apiNodeId, srcNodeId, Weight=weight)

    for o in api.outputs:
        dstNodeId = get_value_node_id(o)
        weight = g.node[dstNodeId][NODE_WEIGHT]
        g.add_edge(dstNodeId, apiNodeId, Weight=weight)


def dep_handler(dep):
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
cost = 0
for i in range(len(longestPath) - 1):
    srcIdx = i
    dstIdx = i + 1
    srcNodeId = longestPath[srcIdx]
    dstNodeId = longestPath[dstIdx]

    print g.edges[srcNodeId][dstNodeId]

    # nodeId in longestPath:
    # node = g.node[nodeId]
    # weight = node[NODE_WEIGHT]
    # if weight:
    #     cost += weight
    # print nodeId, node

print len(longestPath)
print "done"

print "writing graph...",
nx.write_graphml(g, "cprof.graphml")
print "done!"
