#!/usr/bin/env python 

import sys
import json
import subprocess
from pygraphml import Attribute
from pygraphml import Graph
from pygraphml import GraphMLParser

class Node():
    def __init__(self, ID):
        self.ID = ID


class Edge():
    def __init__(self, src, dst, weight):
        self.src = src
        self.dst = dst
        self.weight = weight

class Value(Node):
    def __init__(self, ID, pos, size, allocation_id):
        Node.__init__(self, ID)
        self.pos = pos
        self.size = size
        self.alloc_id = allocation_id
    def add_to_graph(self, g):
        n = g.add_node(self.ID)
        n["size"] = self.size
        n["pos"] = self.pos
        return n


class Allocation(Node):
    def __init__(self, ID, pos, size, ty, addrsp):
        Node.__init__(self, ID)
        self.pos = pos
        self.size = size
    	self.ty = ty
        self.addrsp = addrsp
    def add_to_graph(self, g):
        n = g.add_node(self.ID)
        n["size"] = self.size
        n["pos"] = self.pos
        return n

class DirectedEdge(Edge):
    def __init__(self, src, dst, weight):
        Edge.__init__(self, src, dst, weight)
    def add_to_graph(self, g, nodes):
        g.add_edge(nodes[self.src], nodes[self.dst], directed=True)



Nodes = {}
Values = {}
Allocations = {}

args = sys.argv[1:]

g = Graph()

# Find all allocations and values
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
            newAlloc = Allocation(Id, pos, size, pagetype, as_type)
            Allocations[Id] = newAlloc
            Nodes[Id] = newAlloc.add_to_graph(g)


        elif "val" in j:
            val = j["val"]
            Id = int(val["id"])
            allocation_id = int(val["allocation_id"])
            size = int(val["size"])
            pos = int(val["pos"])
            newValue = Value(Id, pos, size, allocation_id)
            Values[Id] = newValue


## Use dependences to determine connection between allocations
with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)

        if "dep" in j:
            dep = j["dep"]
            srcValId = int(dep["src_id"])
            dstValId = int(dep["dst_id"])
            srcAllocId = Values[srcValId].alloc_id
            dstAllocId = Values[dstValId].alloc_id
            weight = Allocations[srcAllocId].size
            newEdge = DirectedEdge(srcAllocId, dstAllocId, weight)
            newEdge.add_to_graph(g, Nodes)

parser = GraphMLParser()
parser.write(g, "cprof.graphml")