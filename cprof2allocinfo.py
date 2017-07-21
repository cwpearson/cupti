#!/usr/bin/env python 

import sys
import json
import csv
import math

import pycprof

Allocations = {}
Devices = {}
TransfersIn = {}
TransfersOut = {}
Values = {}

def api_handler(a):
    if type(a) != pycprof.API:
        return

    for val_in in a.inputs:
        vin = Values[val_in]
        ain_id = vin.allocation_id
        ain = Allocations[ain_id]
        if vin.size == 0:
            # print "input value size 0"
            vin.size = ain.size
        TransfersOut[ain_id] += [vin.size]

    for val_out in a.outputs:
        vout = Values[val_out]
        aout_id = vout.allocation_id
        aout = Allocations[aout_id]
        if vout.size == 0:
            # print "output value size 0 ->",
            vout.size = aout.size
            # print vout.size
        TransfersIn[aout_id] += [vout.size]

def allocation_handler(a):
    if type(a) != pycprof.Allocation:
        return
    Allocations[a.id_] = a
    TransfersIn[a.id_] = []
    TransfersOut[a.id_] = []

def value_handler(v):
    if type(v) != pycprof.Value:
        return
    if v.id_ in Values:
        print "duplicate value", v.id_, "overwriting..."
    Values[v.id_] = v

# def handler(o):
#     pass

# def api_handler(o):
        # for i in self.inputs:
        #     for o in self.outputs:
        #         if self.name == "cuLaunch":
        #             edgewriter.writerow([i, o, float(Values[o].size), self.symbol, self.symbol])
        #         else:
        #             edgewriter.writerow([i, o, float(Values[o].size), self.name, self.name])
    # pass

pycprof.run_handler(allocation_handler)
print len(Allocations), "allocations found"

pycprof.run_handler(value_handler)
print len(Values), "values found"

pycprof.run_handler(api_handler)

print sum(len(ts) for _,ts in TransfersIn.iteritems()), "xfers in to allocs"
print sum(len(ts) for _,ts in TransfersOut.iteritems()), "xfers out of allocs"

print sum(sum(ts) for _,ts in TransfersIn.iteritems()), "bytes in to allocs"
print sum(sum(ts) for _,ts in TransfersOut.iteritems()), "bytes out of allocs"