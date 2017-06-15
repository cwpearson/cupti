#!/usr/bin/env python 

import sys
import json

class Node():
    def __init__(self, ID):
        self.ID = ID
    def dot_ID(self):
        return str(self.ID)


class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class Value(Node):
    def __init__(self, ID):
        Node.__init__(self, ID)
    def __str__(self):
       return str(self.ID) + ";" 


class Allocation(Node):
    def __init__(self, ID):
        Node.__init__(self, ID)
    def __str__(self):
       return str(self.ID) + ";" 


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
Allocations = {}


def write_header(dotfile):
    header = "digraph graphname {\n"
    dotfile.write(header)

def write_body(dotfile):
    for a in Allocations:
        dotfile.write(str(a))
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

with open(args[0], 'r') as f:
    for line in f:
      j = json.loads(line)
      print j
      if "val" in j:
         val = j["val"]
         ID = val["id"]
         newValue = Value(val["id"])
         Values[ID] = newValue
         Edges += [DottedEdge(newValue.dot_ID(), val["allocation_id"])]

      elif "dep" in j:
         dep = j["dep"]
         Edges += [DirectedEdge(dep["src_id"], dep["dst_id"])]

      elif "allocation" in j:
         alloc = j["allocation"]
         ID = alloc["id"]
         newAllocation = Allocation(ID)
         Allocations[ID] = newAllocation
      else:
         print "Skipping", j


with open("cprof.dot", 'w') as dotfile:
    write_header(dotfile)
    write_body(dotfile)
    write_footer(dotfile)
