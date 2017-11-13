#!/usr/bin/env python 

import sys
import json
import csv
import math

import pycprof

Allocations = {}
Memories = {}
A2D = {}
D2A = {}
A2A = {}
Values = {}

def api_handler(api):
    if type(api) != pycprof.API:
        return

    name = api.functionName
    if name == "cudaLaunch" or "cublas" in name or "cudnn" in name:
        dev = api.device
        for v_id in api.inputs:
            v = Values[v_id]
            a_id = v.allocation_id
            a = Allocations[a_id]
            if v.size == 0:
                v.size = a.size
            if dev not in A2D:
                A2D[dev] = {}
            if a_id not in A2D[dev]:
                A2D[dev][a_id] = []
            A2D[dev][a_id] += [v.size]


        for v_id in api.outputs:
            v = Values[v_id]
            a_id = v.allocation_id
            a = Allocations[a_id]
            if v.size == 0:
                v.size = a.size
            if dev not in D2A:
                D2A[dev] = {}
            if a_id not in D2A[dev]:
                D2A[dev][a_id] = []
            D2A[dev][a_id] += [v.size]

    else:
        for vin_id in api.inputs:
            vin = Values[vin_id]
            ain_id = vin.allocation_id
            if ain_id not in A2A:
                A2A[ain_id] = {}
            for vout_id in api.outputs:
                vout = Values[vout_id]
                aout_id = vout.allocation_id
                if aout_id not in A2A[ain_id]:
                    A2A[ain_id][aout_id] = []
                assert(vin.size == vout.size)
                # print "adding A2A"
                A2A[ain_id][aout_id] += [vin.size]

    

def allocation_handler(a):
    if type(a) != pycprof.Allocation:
        return
    Allocations[a.id_] = a

    loc = a.mem.location
    Id = a.mem.id_
    if loc not in Memories:
        Memories[loc] = {}
    if Id not in Memories[loc]:
        Memories[loc][Id] = {}
    
    Memories[loc][Id][a.id_] = a

def value_handler(v):
    if type(v) != pycprof.Value:
        return
    if v.id_ in Values:
        print "duplicate value", v.id_, "overwriting..."
    Values[v.id_] = v

pycprof.run_handler(allocation_handler)
print len(Allocations), "allocations found"

pycprof.run_handler(value_handler)
print len(Values), "values found"

pycprof.run_handler(api_handler)
print max(len(D2A), len(A2D)), "devices"

pycprof.set_edge_fields(["Weight"])
pycprof.set_node_fields(["pos", "size"])

# Add allocation nodes
for aid, alloc in Allocations.iteritems():
    node = {"pos": alloc.pos, "size": alloc.size}
    pycprof.add_node(aid, node)

# add device nodes
for dev in D2A:
    pycprof.add_node(dev, {})
for dev in A2D:
    pycprof.add_node(dev, {})

# add edges
for dev, xfers in D2A.iteritems():
    for aid, sizes in xfers.iteritems():
        pycprof.add_edge(dev, aid, {"Weight": math.log(sum(sizes),2)})
for dev, xfers in A2D.iteritems():
    for aid, sizes in xfers.iteritems():
        pycprof.add_edge(aid, dev, {"Weight": math.log(sum(sizes),2)})
for src, dsts in A2A.iteritems():
    for dst in dsts:
        pycprof.add_edge(src, dst, {"Weight": math.log(sum(sizes),2)})

pycprof.write_nodes("nodes")
pycprof.write_edges("edges")
