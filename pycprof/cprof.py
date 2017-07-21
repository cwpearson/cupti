""" Utilities for interacting with cprof data files """

import json

from progress import print_progress

DEFAULT_INPUT = "output.cprof"

_handlers = []

def add_handler(f):
    """ Register a handler to run on cprof object """
    global _handlers
    _handlers += [f]

def set_handlers(l):
    _handlers = l

def run_handlers(handler_list, path=None):
    """ Run a list of handlers on a cprof file """
    if not path:
        path = DEFAULT_INPUT

    with open(path, 'r') as input_file:
        # i = 0
        # for i, l in enumerate(input_file):
        #     pass
        # l = i + 1
        # print l

        # print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
        for i, line in enumerate(input_file):
            # print_progress(i, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
            j = json.loads(line)
            if "val" in j:
                obj = Value(j["val"])
            elif "allocation" in j:
                obj = Allocation(j["allocation"])
            elif "api" in j:
                obj = API(j["api"])
            else:
                continue

            for handler_func in handler_list:
                handler_func(obj)

def run_handler(func, path=None):
    return run_handlers([func], path)

def run(path=None):
    """ Run registered handlers on path """
    run_handlers(_handlers, path)

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
        self.mem = Memory(json.loads(j["mem"]))

class API(object):
    def __init__(self, j):
        self.id_ = int(j["id"])
        self.functionName = j["name"]
        self.symbol = j["symbolname"]
        self.device = j["device"]

        inputs = j["inputs"]
        outputs = j["outputs"]
        if inputs == "":
            self.inputs = []
        else:
            self.inputs = [int(x) for x in inputs]
        if outputs == "":
            self.outputs = []
        else:
            self.outputs = [int(x) for x in outputs]

class Memory(object):
    def __init__(self, j):
        self.location = j["loc"]
        self.id_ = j["id"]