#!/usr/bin/env python 

import sys
import json

class Node():
    def __init__(self, ID):
        self.ID = ID

class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

class Value(Node):
    def __init__(self, ID):
        Node.__init__(self, ID)

class Allocation(Node):
    def __init__(self, ID):
        pass

class DirectedEdge(Edge):
    pass

Edges = []
Values = []
Allocations = []


def write_header():
    pass

def write_body():
    pass

def write_footer():
    pass

args = sys.argv[1:]

with open(args[0], 'r') as f:
    for line in f:
      j = json.loads(line)
      print j
      if "val" in j:
         val = j["val"]
      elif "dep" in j:
         dep = j["dep"]
      elif "allocation" in j:
         alloc = j["allocation"]
      else:
         print "Skipping", j


with open("cprof.dot", 'w') as dotfile:
    write_header(dotfile)
    write_body(dotfile)
    write_footer(dotfile)
