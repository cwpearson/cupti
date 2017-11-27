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

allocRefs = {}

xfers = {}


def api_handler(api):
    if type(api) != pycprof.API:
        return

    name = api.functionName
    if name == "cudaLaunch" or "cublas" in name or "cudnn" in name:
        # print name
        for vId in set(api.inputs + api.outputs):
            v = Values[vId]
            a = Allocations[v.allocation_id]
            if a.loc.type != "host":
                allocRefs.setdefault(a, [])
                allocRefs[a] += [api]

    if name == "cudaMemcpy":
        assert len(api.inputs) == 1
        assert len(api.outputs) == 1

        srcValId = api.inputs[0]
        dstValId = api.outputs[0]

        srcVal = Values[srcValId]
        dstVal = Values[dstValId]

        srcAl = Allocations[srcVal.allocation_id]
        dstAl = Allocations[dstVal.allocation_id]

        srcLoc = (srcAl.loc.type, srcAl.loc.id_)
        dstLoc = (dstAl.loc.type, dstAl.loc.id_)

        xfers.setdefault(srcLoc, {})
        xfers[srcLoc].setdefault(dstLoc, [])
        xfers[srcLoc][dstLoc] += [srcAl.size]

    # if name == "cudaLaunch" or "cublas" in name or "cudnn" in name:
    #     dev = api.device
    #     for v_id in api.inputs:
    #         v = Values[v_id]
    #         a_id = v.allocation_id
    #         a = Allocations[a_id]
    #         if v.size == 0:
    #             v.size = a.size
    #         if dev not in A2D:
    #             A2D[dev] = {}
    #         if a_id not in A2D[dev]:
    #             A2D[dev][a_id] = []
    #         A2D[dev][a_id] += [v.size]

    #     for v_id in api.outputs:
    #         v = Values[v_id]
    #         a_id = v.allocation_id
    #         a = Allocations[a_id]
    #         if v.size == 0:
    #             v.size = a.size
    #         if dev not in D2A:
    #             D2A[dev] = {}
    #         if a_id not in D2A[dev]:
    #             D2A[dev][a_id] = []
    #         D2A[dev][a_id] += [v.size]

    # else:
    #     for vin_id in api.inputs:
    #         vin = Values[vin_id]
    #         ain_id = vin.allocation_id
    #         if ain_id not in A2A:
    #             A2A[ain_id] = {}
    #         for vout_id in api.outputs:
    #             vout = Values[vout_id]
    #             aout_id = vout.allocation_id
    #             if aout_id not in A2A[ain_id]:
    #                 A2A[ain_id][aout_id] = []
    #             assert(vin.size == vout.size)
    #             # print "adding A2A"
    #             A2A[ain_id][aout_id] += [vin.size]


def allocation_handler(a):
    if type(a) != pycprof.Allocation:
        return
    Allocations[a.id_] = a

    # loc = a.loc
    # Id = a.mem.id_
    # if loc not in Locations:
    #     Loc[loc] = {}
    # if Id not in Memories[loc]:
    #     Memories[loc][Id] = {}

    # Memories[loc][Id][a.id_] = a


def value_handler(v):
    if type(v) != pycprof.Value:
        return
    if v.id_ in Values:
        print "duplicate value", v.id_, "overwriting..."
    Values[v.id_] = v


pycprof.run_handler(allocation_handler, path=sys.argv[1])
print len(Allocations), "allocations found"

pycprof.run_handler(value_handler, path=sys.argv[1])
print len(Values), "values found"

pycprof.run_handler(api_handler, path=sys.argv[1])
print max(len(D2A), len(A2D)), "devices"

# pycprof.set_edge_fields(["Weight"])
# pycprof.set_node_fields(["pos", "size"])

# Add allocation nodes
# for aid, alloc in Allocations.iteritems():
#     node = {"pos": alloc.pos, "size": alloc.size}
#     pycprof.add_node(aid, node)

# add device nodes
# for dev in D2A:
#     pycprof.add_node(dev, {})
# for dev in A2D:
#     pycprof.add_node(dev, {})

# add edges
# for dev, xfers in D2A.iteritems():
#     for aid, sizes in xfers.iteritems():
#         pycprof.add_edge(dev, aid, {"Weight": math.log(sum(sizes), 2)})
# for dev, xfers in A2D.iteritems():
#     for aid, sizes in xfers.iteritems():
#         pycprof.add_edge(aid, dev, {"Weight": math.log(sum(sizes), 2)})
# for src, dsts in A2A.iteritems():
#     for dst in dsts:
#         pycprof.add_edge(src, dst, {"Weight": math.log(sum(sizes), 2)})

# pycprof.write_nodes("nodes")
# pycprof.write_edges("edges")


def bin_idx(i):
    return int(math.log(i, 2))


maxBin = max(bin_idx(a.size) for a in Allocations.itervalues())

countHistograms = {}
sizeHistograms = {}
xferCountHistograms = {}
xferSizeHistograms = {}

for i, a in Allocations.iteritems():
    loc = (a.loc.type, a.loc.id_)
    # initialize
    countHistograms.setdefault(loc, [0 for i in range(maxBin + 1)])
    sizeHistograms.setdefault(loc, [0 for i in range(maxBin + 1)])
    # fill
    countHistograms[loc][bin_idx(a.size)] += 1
    sizeHistograms[loc][bin_idx(a.size)] += a.size

print "Size Histograms:"
for loc, hist in sizeHistograms.iteritems():
    print loc, hist

# print "Count Histogram"
# print countHistograms


# for v in Values.itervalues():
#     print v.allocation_id

# find max transfer
maxBin = 0
for src, dsts in xfers.iteritems():
    for dst, sizes in dsts.iteritems():
        for size in sizes:
            maxBin = max(maxBin, bin_idx(size))
for src, dsts in xfers.iteritems():
    for dst, sizes in dsts.iteritems():
        for size in sizes:
            binIdx = bin_idx(size)
            xferCountHistograms.setdefault(src, {})
            xferCountHistograms[src].setdefault(
                dst, [0 for i in range(maxBin + 1)])
            xferSizeHistograms.setdefault(src, {})
            xferSizeHistograms[src].setdefault(
                dst, [0 for i in range(maxBin + 1)])
            xferCountHistograms[src][dst][binIdx] += 1
            xferSizeHistograms[src][dst][binIdx] += size

print "Transfer counts"
for src, dsts in xfers.iteritems():
    for dst, sizes in dsts.iteritems():
        print src, dst, xferCountHistograms[src][dst]

print "Transfer totals"
for src, dsts in xfers.iteritems():
    for dst, sizes in dsts.iteritems():
        print src, dst, xferSizeHistograms[src][dst]
