## This file manages the experiments and calculates E values to test based
## on a gradient descent algorithm aimed to minimize error

from scipy.optimize import minimize
from experiment import *
import os
import post_proc

# Function that outputs error from E value
num = 1
def calculate_error(E):
    global num
    # Create experiment object and file to save data
    exp = experiment()
    [exp.E] = E
    os.mkdir('./iteration%i'%num)
    save_experiment('./iteration%i/data.pkl'%num, exp)

    # Print out for viewing purposes
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    print('Step = ', num)
    print('E = ', exp.E)
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    # Run experiment
    post_proc.run_post_proc(num, True)

    # Import back in experiment
    exp = open_experiment('./iteration%i/data.pkl'%num)

    num += 1

    # Return the error
    print(exp.error)
    return exp.error

# Scipy minimizer that actually minimizes error function
minimizer = minimize(calculate_error, 1.0, options={'disp':True, 'eps':0.1})