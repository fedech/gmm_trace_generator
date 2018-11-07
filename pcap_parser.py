#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:01:05 2018

Pcap parser to extract capacity traces from full packet captures (TCP only)

@author: Federico Chiariotti
"""

import pandas as pd
from scapy.all import *
import numpy as np


class Pcap_parser():
    
    def __init__(self, address, incoming, timestep, outfile):
        """
        address: address to trace
        incoming: true to evaluate download capacity, false for upload
        timestep: timestep of the capacity trace
        outfile: output filename
        """
        self.address = address
        self.incoming = incoming
        self.timestep = timestep
        self.outfile = outfile
        
        # Initialization
        self.offset = 0
        self.n_packets = 0
        self.duration = 0
        self.cap_trace = []
        
    # Read a trace and convert it into a capacity trace
    def read_trace(self, tracepath, save):
        """
        tracepath: file path to the pcap trace
        save: true to save the capacity trace
        """
        # Read the trace
        trace = rdpcap(tracepath)
        self.n_packets = len(trace)
        if (len(trace) == 0):
            return
            
        self.offset = trace[0].time
        self.duration = trace[-1].time - self.offset
        first = -1
        last = -1
        
        # Create capacity trace
        self.cap_trace = np.zeros(int(self.duration / self.timestep))
        
        for i in range(0, self.n_packets):
            pkt = trace[i]
            # Discard packets to other addresses
            if ('TCP' in pkt and (pkt[1].dst == self.address and self.incoming) or (pkt[1].src == self.address and not self.incoming)):
                # Find packet slot
                time = pkt.time - self.offset
                index = int(time / self.timestep)
                # Add packet length to capacity
                data = pkt[2].len
                self.cap_trace[index] += data
        if (save):
            self._save_trace()

    # Save a capacity trace in csv format
    def _save_trace(self):
        # Trace format: absolute time, relative time, latitude (empty), longitude (empty), capacity, timestep
        out = f = open(self.outfile,'w')
        for i in range(0, len(self.cap_trace)):
            relative = i * self.timestep
            time = self.offset + relative
            out.write(str(time) + ' ' + str(relative) + ' 0 0 ' + str(int(self.cap_trace[i])) + ' ' + str(self.timestep) + '\n')
        out.close()
      
    
if __name__ == '__main__':
    parser = Pcap_parser('10.0.0.127', False, 0.1, 'bus_1.csv')
    parser.read_trace('bus_1.pcap', True)      
