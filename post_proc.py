## This file contains all of the post processing functions for an individual experimental trial.
## By importing sim as a module, it also runs the physical experiment.

# Other import statements
from scipy import stats, interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from experiment import *
import sim

def run_post_proc (num, run_sim):
    # Runs sim module
    if run_sim:
        sim.run_sim(num, True)

    # Imports experimental data
    exp = open_experiment('./iteration%i/data.pkl'%num)

    ideal_disp = exp.ideal_disp
    E_assign = exp.E
    nu_assigned = exp.nu
    exp_data = exp.exp_data

    
    i = 0
    while i < len(exp_data):
        if exp_data[i,0] > 3:
            break
        i += 1

    exp_data = exp_data[0:i-1,:]

    sim_force = exp.force_array
    sim_disp = exp.u_array

    x_lo = -exp.length/2; x_hi = exp.length/2; y_lo = -exp.height/2; y_hi = exp.height/2; z_lo = -exp.width/2; z_hi = exp.width/2

    def get_mi(P,y):
        b = (z_hi - z_lo)
        h = (y_hi - y_lo)
        I = b * h ** 3.0 / 12.0
        M = P * y 
        M_over_I = M / I
        return abs(M_over_I)

        
    def get_curvature(disp_arr):
        r = x_hi - x_lo; ct = (x_lo + x_hi)/2.0
        y_orig = np.asarray([0.0,0.0,0.0,0.0,0.0])
        x_orig = np.asarray([ct-r/3.0, ct-r/6.0 , ct, ct+r/6.0, ct+ r/3.0])
        idx_keep = [0,1,2,3,4]
        idx_x_keep = [0,3,6,9,12]
        idx_y_keep = [1,4,7,10,13]
        x = x_orig[idx_keep] + disp_arr[idx_x_keep]
        y = y_orig[idx_keep] + disp_arr[idx_y_keep]
        pcoef = np.polyfit(x,y,4)
        pfit  = np.poly1d(pcoef)
        pder1 = np.polyder(pfit,1) #first order derivative
        pder2 = np.polyder(pfit,2) #second order derivative
        yp0  =  pder1(x_orig[2]+disp_arr[6]) #first order derivative evaluated at the center point
        ypp0 =  pder2(x_orig[2]+disp_arr[6]) #second order derivative evaluated at the center point
        
        curvature = ypp0/( 1.0 + yp0**2.0  )**(3.0/2.0)
        plt.plot(x,y,'-o')
        plt.plot(x,pfit(x),'--*')
        
        
        return curvature
        
    mi_list = []; c_list = [] 

    
    num_step = sim_force.shape[0]
    
    
    plt.figure()
    for kk in range(0,num_step):
        P = sim_force[kk,0]
        y = sim_disp[kk,7]*(-1)
        mi_list.append(get_mi(P,y) )
        c_list.append(get_curvature(sim_disp[kk,:]))
    
    plt.axis('equal')
    plt.savefig('./iteration%i/curves'%(num))
    
    plt.figure()
    plt.plot(c_list,mi_list)
    plt.savefig('./iteration%i/graph_curve'%(num))

    plt.figure()
    applied_disp_list = exp.disp_list + ideal_disp*np.ones((len(exp.disp_list)))
    applied_disp  = [] 
    for kk in range(0,len(applied_disp_list)):
        applied_disp.append(-1*applied_disp_list[kk])

    i = 0
    while i < len(applied_disp):
        if applied_disp[i] > 3:
            break
        i += 1

    applied_disp = applied_disp[0:i-1]
    mi_list = mi_list[0:i-1]

    exp.sim_data = np.vstack((applied_disp, mi_list))

    # Finds residuals and error
    sim_func = interpolate.interp1d(applied_disp, mi_list, fill_value='extrapolate')
    residuals = np.zeros(len(applied_disp))
    esiduals = np.array([])
    for i in range(len(exp_data[0,:])):
        residuals = np.append(residuals, exp_data[1,i] - sim_func(exp_data[0,i]))
    exp.error = np.dot(residuals, residuals)
    exp.residuals = residuals

    # Finds R^2 value
    avg = sum(exp_data[1,:]) / len(exp_data[1,:])
    norm_array = exp_data[1,:] - avg
    sstot = np.dot(norm_array, norm_array)
    exp.rsquared = 1 - (exp.error / sstot)
    exp.mu = E_assign / (2*(1 + nu_assigned))
    
    save_experiment('./iteration%i/data.pkl'%num, exp)

    # Graphs moment displacement curve
    plt.scatter(exp_data[:,0],exp_data[:,1], c='r', s=10)
    #plt.plot(applied_disp,mi_list, 'k')
    plt.plot(applied_disp, sim_func(applied_disp), 'k')
    plt.ylabel('M/I (mN*mm^-3)')
    plt.xlabel('Displacement (mm)')
    plt.title('Moment-Displacement Curve @ mu = ' + str(exp.mu))
    plt.savefig('./iteration%i/graph_disp'%(num))

    # Graphs residuals
    plt.clf()
    plt.scatter(applied_disp,residuals, c='r', s=10)
    plt.ylabel('M/I (mN*mm^-3)')
    plt.xlabel('Displacement (mm)')
    plt.title('Moment-Displacement Residuals @ mu = ' + str(mu))
    plt.savefig('./iteration%i/graph_disp_norm'%(num))

