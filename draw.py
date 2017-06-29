#!/usr/bin/env python 

import sys
import json
import subprocess

class Node():
    def __init__(self, ID):
        self.ID = ID
    def dot_Id(self):
        return str(self.ID)
    def dot_label(self):
        return self.dot_Id()


class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class Value(Node):
    def __init__(self, ID, pos, size):
        Node.__init__(self, ID)
        self.size = size
        self.pos = pos
    def dot_shape(self):
        return "record"
    def __str__(self):
       return self.dot_Id() + ' [shape='+self.dot_shape()+',label="'+ self.dot_label() +'" ] ;' 
    def dot_label(self):
        return "{ value: " + self.dot_Id() + \
               " | size: " + str(self.size) + \
               " | pos: "  + str(self.pos) + \
               " } "

num_subgraphs=0
class Subgraph():
    def __init__(self, label):
        self.label = label
        self.nodes = {}
        global num_subgraphs
        self.name = "cluster_"+str(num_subgraphs)
        num_subgraphs += 1

class Location(Subgraph):
    def __init__(self, location):
        Subgraph.__init__(self, location)
        self.allocation_ids = []
    def __str__(self):
        s = "subgraph " + self.name + " {\n"
        s += "style=filled;\n"
        s += "color=lightgrey;\n"
        s += 'label = "' + self.label+ '";\n'
        for Id in self.allocation_ids:
            s += str(Allocations[Id]) + '\n'
        s += "}"
        return s

class Allocation(Subgraph):
    def __init__(self, Id, pos, size):
        Subgraph.__init__(self, Id)
        self.value_ids = []
        self.pos = pos
        self.size = size
    def __str__(self):
        s = "subgraph " + self.name + " {\n"
        s += "style=filled;\n"
        s += "color=grey;\n"
        s += 'label = "id: ' + self.label + "\n" + \
                      "pos: " + self.pos + "\n" + \
                      "size: " + self.size + '";\n'
        for Id in self.value_ids:
            s += str(Id) + ';\n'
        s += "}"
        return s

class DirectedEdge(Edge):
    def __init__(self, src, dst):
        Edge.__init__(self, src, dst)
    def __str__(self):
       return str(self.src) + " -> " + str(self.dst) + ";" 

class DottedEdge(Edge):
    def __init__(self, src, dst):
        Edge.__init__(self, src, dst)
    def __str__(self):
       return str(self.src) + " -> " + str(self.dst) + " [dir=none, style=dashed];" 

Edges = []
Values = {}
Locations = {}
Allocations = {}

def write_header(dotfile):
    header = "digraph graphname {\n"
    dotfile.write(header)

def write_body(dotfile):
    for l in Locations:
        dotfile.write(str(Locations[l]))
        dotfile.write("\n")
    for k in Values:
        dotfile.write(str(Values[k]))
        dotfile.write("\n")
    for e in Edges:
	dotfile.write(str(e))
        dotfile.write("\n")

def write_footer(dotfile):
    header = "}\n"
    dotfile.write(header)

args = sys.argv[1:]

## first pass - set up allocations
with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)
        if "allocation" in j:
            alloc = j["allocation"]
            Id = alloc["id"]
            size = alloc["size"]
            pos = alloc["pos"]
            
            loc = alloc["loc"]
            if loc not in Locations:
                print "adding new location", loc
                Locations[loc] = Location(loc)

            print "adding allocation", Id, "to", loc
            Locations[loc].allocation_ids += [Id]
            newAllocation = Allocation(Id, pos, size)
            Allocations[Id] = newAllocation

## second pass - set up values and dependences
with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)

        if "val" in j:
            val = j["val"]
            Id = val["id"]
            allocation_id = val["allocation_id"]
            size = val["size"]
            pos = val["pos"]
            newValue = Value(Id, pos, size)
            print "adding value", Id, "to alloc", allocation_id
            Allocations[allocation_id].value_ids += [Id]

            Values[Id] = newValue

        elif "dep" in j:
            dep = j["dep"]
            Edges += [DirectedEdge(dep["src_id"], dep["dst_id"])]

with open("cprof.dot", 'w') as dotfile:
    write_header(dotfile)
    write_body(dotfile)
    write_footer(dotfile)

print subprocess.check_output(['dot','-Tpdf', '-o', 'cprof.pdf', 'cprof.dot'])