import csv

_edgeFilePath = "edges.csv"
_nodeFilePath = "nodes.csv"

_edgeFieldList = []
_nodeFieldList = []

_edges = {}
_nodes = {}

def set_edge_fields(fieldList):
    global _edgeFieldList
    _edgeFieldList = fieldList

def set_node_fields(fieldList):
    global _nodeFieldList
    _nodeFieldList = fieldList

def add_node(Id, node):
    global _nodes
    _nodes[Id] = node

def add_edge(src, dst, edge):
    global _edges
    if src not in _edges:
        _edges[src] = {}
    _edges[src][dst] = edge

def write_nodes(basename):
    with open(basename + ".csv", 'wb') as f:
        writer = csv.writer(f, delimiter=",", quotechar='|')
        writer.writerow(["id"] + _nodeFieldList)
        for nid, n in _nodes.iteritems():
            writer.writerow([nid] + [n[field] if field in n else "" for field in _nodeFieldList])

def write_edges(basename):
    with open(basename + ".csv", 'wb') as f:
        writer = csv.writer(f, delimiter=",", quotechar='|')
        writer.writerow(["Source", "Target"] + _edgeFieldList)
        for src, dsts in _edges.iteritems():
            for dst, e in dsts.iteritems():
                writer.writerow([src, dst] + [e[field] if field in e else "" for field in _edgeFieldList])