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

    if a.functionName == "cuLaunch":
        pass
    else:
        for val_in in a.inputs:
            for val_out in a.outputs:
                vin = Values[val_in]
                vout = Values[val_out]
                if vin.size != vout.size:
                    print vin.size, vout.size, a.functionName
                ain = Allocations[vin.allocation_id]
                aout = Allocations[vout.allocation_id]

                if ain not in Transfers:
                    Transfers[ain] = {}
                if aout not in Transfers[ain]:
                    Transfers[ain][aout] = []
                Transfers[ain][aout] += [vin.size]


    # print "value_handler called"
    # writer.writerow([v.ID, v.size, v.pos])

def allocation_handler(a):
    if type(a) != pycprof.Allocation:
        return
    Allocations[a.id_] = a

def value_handler(v):
    if type(v) != pycprof.Value:
        return
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


args = sys.argv[1:]

# Find all allocations and values

with open('edges.csv', 'wb') as edgefile:
    with open('nodes.csv', 'wb') as nodefile:
        edgewriter = csv.writer(edgefile, delimiter=',')
        nodewriter = csv.writer(nodefile, delimiter=',')
        nodewriter.writerow(["id", "size", "pos"])
        edgewriter.writerow(["source", "target", "weight", "Label", "name"])
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
                    if 0 == size:
                        size = Allocations[allocation_id].size
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
