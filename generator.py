#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 22:54:23 2018

Trace generator using the Gaussian mixture model

@author: Federico Chiariotti
"""

import os
import csv

class TraceGenerator():
    
    def __init__(self, n_features, n_components):
        self.set_parameters(n_features, n_components)
    
    # Load a trace file
    def load_trace(self, trace, address, incoming, timestep):
        tmpfile = './tmp.csv'
        self.parser = Pcap_parser(address, incoming, timestep, tmpfile)
        self.parser.read_trace(trace, True)     
        load_capacity(tmpfile)
        os.remove(tmpfile)
        
    # Create the model object
    def _load_model(self):
        self.model = GMM_Channel(self.n_features, self.n_components)
        
    # Set the statistical parameters of the GMM generator
    def set_parameters(self, n_features, n_components):
        self.n_features = n_features
        self.n_components = n_components
        _load_model()
    
    # Load a capacity file and train the model
    def load_capacity(self, filepath)
        single_df = pd.read_csv(filepath, sep=' ', names=['timestamp', 'time', 'lat', 'long', 'bytes', 'period'])
        capacity = np.array(single_df['bytes'].values * 8, dtype=np.float)
        self.model.fit(capacity)
        
    # Generate a trace using the current file
    def generate_trace(self, length, seed, outfile):
        gen_trace = model.sample(length, random_seed=seed)
        with open(outfile + '.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, length):
                writer.writerow(gen_trace[i])
        
    # Save the trained model
    def save_model(self, outfolder, outfile):
        model.save_model(outfolder / outfile)
        
#TODO write main function
