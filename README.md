# How to use experimental package

## Formatting and importing experimental data
Experimental data should be saved in an excel file located in the same folder as the other files. It's name should be specified in experiment.py. This file should contain only the raw data with the displacement values in the first column and the M/I values in the second column.

## Setting initial parameters
Initial parameters need to be set in the experiment.py file. These include the following: 
1. The initial dimensions of the hydrogel
2. The value of Poisson's ratio for the linear elasticity approximation
3. The initial curvature of the hydrogel caused by body forces
4. The file name of the experimental data

## Executing program


