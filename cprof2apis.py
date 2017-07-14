#!/usr/bin/env python 

import sys
import json
import csv

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
    def write_to_csv(self, writer):
        writer.writerow([self.ID, self.size, self.pos])


class Allocation(Node):
    def __init__(self, ID, pos, size, ty, addrsp):
        Node.__init__(self, ID)
        self.pos = pos
        self.size = size
    	self.ty = ty
        self.addrsp = addrsp

class API(Node):
    def __init__(self, ID, inputs, outputs, name, symbol):
        Node.__init__(self, ID)
        self.name = name
        self.symbol = symbol
        if inputs == "":
            self.inputs = []
        else:
            self.inputs = [int(x) for x in inputs]
        if outputs == "":
            self.outputs = []
        else:
            self.outputs = [int(x) for x in outputs]
    def write_to_csv(self, nodewriter, edgewriter):
        nodewriter.writerow([self.ID, 0, 0, self.name, self.symbol])
        for i in self.inputs:
            edgewriter.writerow([i, self.ID, float(Values[i].size)])
        for o in self.outputs:
            edgewriter.writerow([self.ID, o, float(Values[o].size)])

class DirectedEdge(Edge):
    def __init__(self, src, dst, weight):
        Edge.__init__(self, src, dst, weight)



Nodes = {}
Values = {}
Allocations = {}

args = sys.argv[1:]

# Find all allocations and values

with open('edges.csv', 'wb') as edgefile:
    with open('nodes.csv', 'wb') as nodefile:
        edgewriter = csv.writer(edgefile, delimiter=',')
        nodewriter = csv.writer(nodefile, delimiter=',')
        nodewriter.writerow(["id", "size", "pos", "name", "symbol"])
        edgewriter.writerow(["source", "target", "weight"])
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


                elif "val" in j:
                    val = j["val"]
                    Id = int(val["id"])
                    allocation_id = int(val["allocation_id"])
                    size = int(val["size"])
                    pos = int(val["pos"])
                    newValue = Value(Id, pos, size, allocation_id)
                    Values[Id] = newValue
                    n = newValue.write_to_csv(nodewriter)
                    Nodes[Id] = n

    ## Draw API calls
        with open(args[0], 'r') as f:
            for line in f:
                j = json.loads(line)

                if "api" in j:
                    api = j["api"]
                    Id = int(api["id"])
                    inputIds = api["inputs"]
                    outputIds = api["outputs"]
                    name = api["name"]
                    symbol = api["symbolname"]
                    newApi = API(Id, inputIds, outputIds, name, symbol)
                    newApi.write_to_csv(nodewriter, edgewriter)
