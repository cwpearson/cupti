#!/usr/bin/env python 

import sys
import json
import csv
import math

import pycprof

Allocations = {}
Devices = {}
Transfers = {}
Values = {}

def api_handler(a):
    if type(a) != pycprof.API:
        return

    if a.functionName == "cudaLaunch":
        return
    if "cudnn" in a.functionName:
        return
    if "cublas" in a.functionName:
        return
    
    for val_in in a.inputs:
        for val_out in a.outputs:
            vin = Values[val_in]
            vout = Values[val_out]
            ain_id = vin.allocation_id
            aout_id = vout.allocation_id
            ain = Allocations[ain_id]
            aout = Allocations[aout_id]


            if vin.size == 0:
                vin.size = ain.size
            if vout.size == 0:
                vout.size = aout.size

            if vin.size != vout.size:
                print vin.size, vout.size, a.functionName

            if ain_id not in Transfers:
                Transfers[ain_id] = {}
            if aout_id not in Transfers[ain_id]:
                Transfers[ain_id][aout_id] = []
            Transfers[ain_id][aout_id] += [vin.size]


    # print "value_handler called"
    # writer.writerow([v.ID, v.size, v.pos])

def allocation_handler(a):
    if type(a) != pycprof.Allocation:
        return
    Allocations[a.id_] = a

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

pycprof.run_handler(value_handler)
print len(Values), "values found"

pycprof.run_handler(allocation_handler)
print len(Allocations), "allocations found"

pycprof.run_handler(api_handler)
