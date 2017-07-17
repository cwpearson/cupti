import json

_handlers = []

def register_handler(f):
    """ Register a handler to run on cprof object """
    global _handlers
    _handlers += [f]


def run(path=None):
    """ Run registered handlers on path """
    if not path:
        path = "output.cprof"

    with open(path, 'r') as f:
        for line in f:
            j = json.loads(line)
            if "value" in j:
                o = Value(j["value"])
            elif "allocation" in j:
                o = Allocation(j["allocation"])

            for h in _handlers:
                h(o)

class Value(object):
    def __init__(self, j):
        self.id_ = int(j["id"])
        self.size = int(j["size"])
        self.pos = int(j["pos"])
        self.allocation_id = int(j["allocation_id"])
        self.initialized = j["initialized"]

class Allocation(object):
    def __init__(self, j):
        self.id_ = int(j["id"])
        self.size = int(j["size"])
        self.pos = int(j["pos"])
        self.type = j["type"]
        self.address_space = json.loads(j["addrsp"])
        self.mem = json.loads(j["mem"])

class API(object):
    def __init__(self, j):
        self.id_ = int(j["id"])
        self.name = j["name"]
        self.symbol = j["symbol"]

        inputs = json.loads(j["inputs"])
        outputs = json.loads(j["outputs"])
        if inputs == "":
            self.inputs = []
        else:
            self.inputs = [int(x) for x in inputs]
        if outputs == "":
            self.outputs = []
        else:
            self.outputs = [int(x) for x in outputs]