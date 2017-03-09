"""

  A simple profiler that measures the time for each events.

"""

import torch
import time
import utils

class SimpleProfiler(object):
    def __init__(self):
        self.clocks = {}
        self.times = {}

    def reset(self, event):
        if not event == None:
            self.clocks[event] = None
            self.times[event] = None
        else:
            self.clocks = {}
            self.times = {}

    def start(self, event):
        self.clocks[event] = time.time()
        if self.times[event] == None:
            self.times[event] = 0.0

    def pause(self, event):
        if not self.times[event] ==  None:
            self.times[event] = self.times[event] + time.time() - self.clocks[event]
            self.clocks[event] = 0.0

    def get_time(self, event):
        if not self.times == None:
            return self.times[event]
        else:
            return 0.0

    def print_all(self):
        print "------------ profiler -------------"
        for event in self.times.keys():
            print "Event ", event, "time ", self.times[event]

