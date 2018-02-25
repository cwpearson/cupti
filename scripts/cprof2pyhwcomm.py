#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

import networkx as nx

from pycprof.profile import Profile
from pyhwcomm.machines.minsky import Minsky
from pyhwcomm.parsers import MakeConcrete
from pyhwcomm.executor import ReplayExecutor
import pyhwcomm


parser = argparse.ArgumentParser(description=u'Test script')
parser.add_argument(u"profile_path", type=str,
                    help=u"path to the profile file")
parser.add_argument(
    u'-n', u'--num_lines', type=int, default=0, help=u'the number of lines from the profile to read')
args = parser.parse_args()


print("reading profile...")
profile = Profile(args.profile_path, num_lines=args.num_lines)
print("done")
machine = Minsky()
print("making concrete program...")
program = MakeConcrete(profile, machine)
print("done")
executor = ReplayExecutor()
time = executor(program, machine)
print("Final time:", time)
machine.gpu_speedup = 1.5
time = executor(program, machine)
print("Final time:", time)
