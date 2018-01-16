#!/usr/bin/env python

import sys
import json
import subprocess
from pygraphml import Attribute
from pygraphml import Graph
from pygraphml import GraphMLParser

import pycprof

g = Graph()


def get_node_id(n):
    if type(n) == pycprof.Value:
        return "val" + str(n.id_)
    elif type(n) == pycprof.API:
        return "api" + str(n.id_)
    else:
        print type(n)
        assert False


# Add a node for each value
Values = {}
ValueNodes = {}


def value_handler(val):
    if type(val) != pycprof.Value:
        return
    Values[val.id_] = val

    valNode = g.add_node(get_node_id(val))
    valNode["Weight"] = val.size
    ValueNodes[val.id_] = valNode

# Add a node for each api. Connect values to apis and vis versa


def api_handler(api):
    if type(api) != pycprof.API:
        return
    apiNode = g.add_node(get_node_id(api))
    for i in api.inputs:
        srcNode = ValueNodes[i]
        e = g.add_edge(srcNode, apiNode, directed=True)
        e["Weight"] = float(srcNode["Weight"])
    for o in api.outputs:
        dstNode = ValueNodes[o]
        e = g.add_edge(apiNode, dstNode, directed=True)
        e["Weight"] = float(dstNode["Weight"])


def dep_handler(dep):
    if type(dep) != pycprof.Dependence:
        return
    src = dep.src
    dst = dep.dst
    srcVal = Values[src]
    dstVal = Values[dst]
    # g.add_edge(srcNode, dstNode, directed=True)


pycprof.run_handler(value_handler, path=sys.argv[1])
pycprof.run_handler(api_handler, path=sys.argv[1])
pycprof.run_handler(dep_handler, path=sys.argv[1])

# create nodes for compute
# pycprof.run_handler(api_handler)

# create nodes for storage
# pycprof.run_handler(dep_handler, path=sys.argv[1])


print "writing graph...",
parser = GraphMLParser()
parser.write(g, "cprof.graphml")
print "done!"

'''
class Node():
    def __init__(self, ID):
        self.ID = ID


class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class Value(Node):
    def __init__(self, ID, pos, size, allocation_id):
        Node.__init__(self, ID)
        self.pos = pos
        self.size = size
        self.allocation_id = allocation_id

    def add_to_graph(self, g):
        n = g.add_node(self.ID)
            += Attribute("size", self.size, type='int')
        n["pos"] = self.pos
        return n


class Allocation():
    def __init__(self, Id, pos, size, ty, addrsp):
        self.Id = Id
        self.pos = pos
        self.size = size
        self.ty = ty
        self.addrsp = addrsp


class DirectedEdge(Edge):
    def __init__(self, src, dst):
        Edge.__init__(self, src, dst)

    def add_to_graph(self, g, nodes):
        g.add_edge(nodes[self.src], nodes[self.dst], directed=True)


Nodes = {}
Allocations = {}

args = sys.argv[1:]

g = Graph()

with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)
        if "allocation" in j:
            alloc = j["allocation"]
            Id = int(alloc["id"])
            pos = int(alloc["pos"])
            size = int(alloc["size"])
            AS = json.loads(alloc["addrsp"])
            AM = json.loads(alloc["mem"])
            pagetype = alloc["type"]
            as_type = AS["type"]

            Allocations[Id] = Allocation(Id, pos, size, pagetype, as_type)

# second pass - set up values and dependences
with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)

        if "val" in j:
            val = j["val"]
            Id = int(val["id"])
            allocation_id = int(val["allocation_id"])
            size = int(val["size"])
            pos = int(val["pos"])
            newValue = Value(Id, pos, size, allocation_id)
            Nodes[Id] = newValue.add_to_graph(g)

        elif "dep" in j:
            dep = j["dep"]
            srcId = int(dep["src_id"])
            dstId = int(dep["dst_id"])
            newEdge = DirectedEdge(srcId, dstId)
            newEdge.add_to_graph(g, Nodes)

parser = GraphMLParser()
parser.write(g, "cprof.graphml")

'''
