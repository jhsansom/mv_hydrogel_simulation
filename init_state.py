## This file calculates the starting displacement and body force based on the initial curvature
## of the physical experiment. 

# import statements
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *
import numpy as np
from scipy import optimize
import pandas as pd
from experiment import *

def run_init_state(num):
    # Imports experimental data
    exp = open_experiment('./iteration%i/data.pkl'%num)

    # assign body force and E
    body_force = exp.body_force
    E_assign = exp.E
    nu_assigned = exp.nu
    initial_curvature = exp.initial_curvature

    # starting parameters
    num_nodes = 5
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = 2

    # define mesh
    x_lo = -exp.length/2; x_hi = exp.length/2; y_lo = -exp.height/2; y_hi = exp.height/2; z_lo = -exp.width/2; z_hi = exp.width/2
    mesh = BoxMesh(Point(x_lo,y_lo,z_lo),Point(x_hi,y_hi,z_hi),20,2,10)

    inertia = ((x_hi - x_lo)**2 + (y_hi - y_lo)**2) / 2

    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    # boundary conditions
    bcPin_line_left  =  CompiledSubDomain("near(x[0], sideX) && near(x[1],sideY)", sideX = x_lo,sideY=0) #y_lo <-- to pin on bottom instead of center
    bcPin_line_right =  CompiledSubDomain("near(x[0], sideX) && near(x[1],sideY)", sideX = x_hi,sideY=0) #y_lo <-- to pin on bottom instead of center 
    bcDom_xLo = CompiledSubDomain("near(x[0], side) ", side = x_lo)
    # -- right boundary, whole side area
    bcDom_xHi = CompiledSubDomain("near(x[0], side)", side = x_hi)

    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    bcDom_xHi.mark(boundary_markers, 1) # Prescribed traction
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # define finite element problem
    up = Function(W)
    (u, p) = split(up)
    dup = TrialFunction(W)
    vq = TestFunction(W)

    T  = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary
    #B  = Constant((0.0, -0.01, 0.0))  # Body force per unit volume
    # --> traction integral 
    integrals_N = [dot(T,u)*ds(1)] # just = 0 here 

    #################################
    # find correct body force
    #################################

    def get_functions():
        return up, dup, vq

    B = Constant((0.0, 0.0, 0.0))
    def f1(body_force):
        global B
        B = Constant((0.0, -body_force, 0.0))  # Body force per unit volume
        print('BODY FORCE = ', body_force)
        # optimize displacement for given body force using newton solver
        disp = optimize.newton(f2, 0.0)

        # calculate curvature
        up, dup, vq = get_functions()
        up, dup, vq, f_int, f_ext = problem_solve(disp, up,dup,vq)
        (u,p) = up.split(True)
        curve = get_curvature(u)

        # return curvature - initial experimental curvature
        print('CURVATURE = ', curve)
        return curve - initial_curvature

    ideal_disp = 0.0
    def f2(disp):
        global ideal_disp
        ideal_disp = disp
        print('DISP = ', disp)
        # displace mesh by given disp
        up, dup, vq = get_functions()
        up, dup, vq, f_int, f_ext = problem_solve(disp, up,dup,vq)
        (u,p) = up.split(True)
        print('Hello')

        # calculate and return force
        return get_rxn_force(W, f_int, f_ext, disp)

    ##########################################################################################
    def problem_solve(applied_disp, up,dup,vq):
        (u, p) = split(up)
        ######################################################################################
        # boundary conditions (inside solver because they change) 
        ######################################################################################
        bcXL = DirichletBC(W.sub(0).sub(0), Constant((0.0)), bcPin_line_left,method="pointwise")
        bcYL = DirichletBC(W.sub(0).sub(1), Constant((0.0)), bcPin_line_left,method="pointwise")
        bcZL = DirichletBC(W.sub(0).sub(2), Constant((0.0)), bcPin_line_left,method="pointwise")
        bcXR = DirichletBC(W.sub(0).sub(0), Constant((applied_disp)), bcPin_line_right,method="pointwise")
        bcYR = DirichletBC(W.sub(0).sub(1), Constant((0.0)), bcPin_line_right,method="pointwise")
        bcZR = DirichletBC(W.sub(0).sub(2), Constant((0.0)), bcPin_line_right,method="pointwise")
        bcs = [bcXL,bcYL,bcZL,bcXR,bcYR,bcZR]
        ######################################################################################
        # define strain energy 
        ######################################################################################
        # Kinematics
        d = len(u)
        I = variable(Identity(d))             # Identity tensor
        F = variable(I + grad(u))             # Deformation gradient
        C = variable(F.T*F)                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ii   = tr(C)
        Iii  = 1/2*(tr(C) - tr(dot(C,C)))
        Iiii = det(C)
        J = det(F)

        # Elasticity parameters
        E, nu = E_assign, nu_assigned 
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        psi = mu/2*(Ii - 3) - (mu + p)*ln(J) - 1/(2*lmbda)*p**2
        
        ######################################################################################
        # set up eqn to solve and solve it   
        ######################################################################################
        f_int = derivative(psi*dx,up,vq)
        f_ext = derivative( dot(B, u)*dx + sum(integrals_N) , up, vq)
        F = f_int - f_ext 
        # Tangent 
        dF = derivative(F, up, dup)
        solve(F == 0, up, bcs, J=dF)
        return up, dup, vq, f_int, f_ext 

    ##########################################################################################
    # functions to post-process the simulations 
    ##########################################################################################

    def get_rxn_force(W, f_int, f_ext, applied_disp):
        x_dofs = W.sub(0).sub(0).dofmap().dofs()
        y_dofs = W.sub(0).sub(1).dofmap().dofs()
        z_dofs = W.sub(0).sub(2).dofmap().dofs()
        
        f_ext_known = assemble(f_ext)
        f_ext_unknown = assemble(f_int) - f_ext_known
        dof_coords = W.tabulate_dof_coordinates().reshape((-1, 3))

        #x_val_min = np.min(dof_coords[:,0]) + 10E-5; x_val_max = np.max(dof_coords[:,0]) - 10E-5
        x_val_min = x_lo
        x_val_max = x_hi

        y_val_min = np.min(dof_coords[:,1]) + 10E-5; y_val_max = np.max(dof_coords[:,1]) - 10E-5

        x_min = []; x_max = [] 
        for kk in x_dofs:
            if near(dof_coords[kk,0], x_val_min) and near(dof_coords[kk,1], ((y_lo + y_hi) / 2)):
                x_min.append(kk)
            if near(dof_coords[kk,0], x_val_max) and near(dof_coords[kk,1], ((y_lo + y_hi) / 2)):
                x_max.append(kk)
        f_sum_left_x = np.sum(f_ext_unknown[x_min])
        f_sum_right_x = np.sum(f_ext_unknown[x_max])		

        y_min = []; y_max = [] 
        for kk in y_dofs:
            if dof_coords[kk,0] < x_val_min and dof_coords[kk,1] < y_val_min: #FLAG <-- update this
                y_min.append(kk)
            if dof_coords[kk,0] > x_val_max and dof_coords[kk,1] < y_val_min: #FLAG <-- update this
                y_max.append(kk)
        f_sum_left_y = np.sum(f_ext_unknown[y_min])
        f_sum_right_y = np.sum(f_ext_unknown[y_max])		

        z_min = []; z_max = [] 
        for kk in z_dofs:
            if dof_coords[kk,0] < x_val_min and dof_coords[kk,1] < y_val_min: #FLAG <-- update this
                z_min.append(kk)
            if dof_coords[kk,0] > x_val_max and dof_coords[kk,1] < y_val_min: #FLAG <-- update this
                z_max.append(kk)
        f_sum_left_z = np.sum(f_ext_unknown[z_min])
        f_sum_right_z = np.sum(f_ext_unknown[z_max])		

        print("x_left, x_right rxn force:", f_sum_left_x, f_sum_right_x)
        print("y_left, y_right rxn force:", f_sum_left_y, f_sum_right_y)
        print("z_left, z_right rxn force:", f_sum_left_z, f_sum_right_z)

        return f_sum_left_x

    # starting nodes
    x_interval = np.linspace(x_lo, x_hi, num_nodes+2)
    coor = np.empty((num_nodes, 3))
    i = 0
    while i < len(coor):
        coor[i, :] = np.array([x_interval[i+1], 0.0, 0.0])
        i += 1

    def get_fiducial(u, coor):
        return np.array([u(coor[0]), u(coor[1]), u(coor[2]), u(coor[3]), u(coor[4])])

    def get_curvature(u):
        nodes = get_fiducial(u, coor) + coor

        x = nodes[:, 0]
        y = nodes[:, 1]

        x = x.flatten()
        y = y.flatten()
        graph = np.polyfit(x, y, 4) 
        y2 = np.poly1d(graph)
        x2 = np.linspace(coor[1,0], coor[3,0], 500)

        ploty = y2(x2)
        plt.clf()
        plt.plot(x2, ploty)
        plt.scatter(x,y)
        plt.xlabel('X - Coordinate')
        plt.ylabel('Y - Coordinate')
        plt.savefig('./iteration%i/nodes.png'%(num))

        graph_prime = np.poly1d.deriv(y2)
        y3 = np.poly1d(graph_prime)

        graph_prime_prime = np.poly1d.deriv(y3)
        y4 = np.poly1d(graph_prime_prime)

        curvature = abs(y4(x2) / ((1 + (y3(x2)) ** 2) ** (3 / 2)))

        return max(curvature)
        

    ####################################################
    # run actual code
    ####################################################

    exp.body_force = optimize.newton(f1, 0.0001)
    exp.ideal_disp = ideal_disp
    save_experiment('./iteration%i/data.pkl'%num, exp)