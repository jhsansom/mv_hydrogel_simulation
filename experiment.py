## This document contains a class with all the data for one experiment iteration
## along with a pickling method to save this data in a file.

import pandas as pd
import numpy as np
import pickle
import datetime
import os

# Input parameters
nu = 0.499
length = 11.78
width = 5.5
height = 1.1
initial_curvature = 0.070037
exp_data_filename = 'data.xlsx'

def save_experiment(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def open_experiment(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class experiment ():
    def __init__ (self):
        # Time for keeping track
        self.time = datetime.datetime.now()

        # Inputs
        self.E = None
        self.nu = nu
        self.mu = None
        self.initial_curvature = initial_curvature
        self.exp_data = self.import_experimental_data(exp_data_filename)
        self.length = length
        self.width = width
        self.height = height

        # Outputs of init_state.py
        self.body_force = None
        self.ideal_disp = None

        # Outputs of sim.py
        self.disp_list = np.array([])
        self.force_array = np.array([])
        self.u_array = np.array([])

        # Outputs of post_proc.py
        self.error = None
        self.rsquared = None

    def import_experimental_data(self, exp_data_filename):
        return np.transpose(np.array(pd.read_excel(exp_data_filename)))
    



