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
    def __init__(self, dot_label):
        global num_subgraphs
        self.name = "cluster_"+str(num_subgraphs)
        self.dot_label_ = dot_label
        num_subgraphs += 1
    ## FIXME: how to say this must be overridden
    def dot_label(self):
        return self.dot_label_


class Memory(Subgraph):
    def __init__(self, ty):
        self.ty = ty
        Subgraph.__init__(self, str(self.ty))
        self.regions = {}
    def __str__(self):
        s = "subgraph " + self.name + " {\n"
        s += "style=filled;\n"
        s += "color=lightgrey;\n"
        s += 'label = "' + self.dot_label()+ '";\n'
        for region_id in self.regions:
            s += str(self.regions[region_id]) + '\n'
        s += "}"
        return s

class Region(Subgraph):
    def __init__(self, Id):
        self.Id = Id
        Subgraph.__init__(self, str(self.Id))
        self.allocations = []
    def __str__(self):
        s = "subgraph " + self.name + " {\n"
        s += "style=filled;\n"
        s += "color=lightgrey;\n"
        s += 'label = "' + self.dot_label()+ '";\n'
        for Id in self.allocations:
            s += str(Allocations[Id]) + '\n'
        s += "}"
        return s


class Allocation(Subgraph):
    def __init__(self, Id, pos, size, ty):
        Subgraph.__init__(self, Id)
        self.value_ids = []
        self.pos = pos
        self.size = size
	self.ty = ty
    def __str__(self):
        s = "subgraph " + self.name + " {\n"
        s += "style=filled;\n"
        s += "color=grey;\n"
        s += 'label = "id: ' + self.dot_label() + "\n" + \
                      "pos: " + self.pos + "\n" + \
                      "size: " + self.size + "\n" + \
                      self.ty + '";\n'
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
Locations = []
Allocations = {}
Memories = {}

def write_header(dotfile):
    header = "digraph graphname {\n"
    #header += "newrank=true\n"
    dotfile.write(header)

def write_body(dotfile):
    for m in Memories:
        dotfile.write(str(Memories[m]))
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

## first pass - handle allocation statements
with open(args[0], 'r') as f:
    for line in f:
        j = json.loads(line)
        if "allocation" in j:
            alloc = j["allocation"]
            Id = alloc["id"]
            size = alloc["size"]
            pos = alloc["pos"]
            ty = alloc["type"]
            mem = json.loads(alloc["mem"])

            mem_loc = mem["loc"]
            mem_id = mem["id"]
            if mem_loc not in Memories:
                print "found new mem_loc", mem_loc
                Memories[mem_loc] = Memory(mem_loc)
            if mem_id not in Memories[mem_loc].regions:
                print "found new mem_id", mem_id
                Memories[mem_type].regions[mem_id] = Region(mem_id)


            print "adding allocation", Id
            Memories[mem_type].regions[mem_id].allocations += [Id]
            newAllocation = Allocation(Id, pos, size, ty)
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
