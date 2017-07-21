import csv

_edgeFilePath = "edges.csv"
_nodeFilePath = "nodes.csv"

_edgeFieldList = []
_nodeFieldList = []

_edges = {}
_nodes = {}

def set_edge_fields(fieldList):
    _edgeFieldList = fieldList
    pass

def set_node_fields(fieldList):
    _nodeFieldList = fieldList
    pass

def write_nodes(basename):
    with open(basename + ".csv") as f:
        writer = csv.writer(f, delimiter=",", quotechar='|')
        writer.writerow(_nodeFieldList)
        for n in _nodes:
            writer.writerow([n[field] for field in _nodeFieldList])

def write_edges(basename):
    with open(basename + ".csv") as f:
        writer = csv.writer(f, delimiter=",", quotechar='|')
        writer.writerow(_edgeFieldList)
        for e in _edges:
            writer.writerow([e[field] for field in _edgeFieldList])