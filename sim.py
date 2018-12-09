## This file actually executes the physical experiment when imported by post_proc.py
## It also imports init_state.py, which calculates the intial body force to use

# Other import statements
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dolfin import *
from mshr import *
import numpy as np
from experiment import *
import init_state

def run_sim(num, run_init):
	if run_init:
		init_state.run_init_state(num)

	# Imports experimental data
	exp = open_experiment('./iteration%i/data.pkl'%num)

	# assign body force and E
	body_force = exp.body_force
	E_assign = exp.E
	nu_assigned = exp.nu

	# Compiler settings
	parameters["form_compiler"]["cpp_optimize"] = True
	parameters["form_compiler"]["representation"] = "uflacs"
	parameters["form_compiler"]["quadrature_degree"] = 2

	# Define mesh
	x_lo = -exp.length/2; x_hi = exp.length/2; y_lo = -exp.height/2; y_hi = exp.height/2; z_lo = -exp.width/2; z_hi = exp.width/2
	mesh = BoxMesh(Point(x_lo,y_lo,z_lo),Point(x_hi,y_hi,z_hi),20,2,10)

	# Define function space
	P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
	P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
	TH = P2 * P1
	W = FunctionSpace(mesh, TH)

	# Define boundaries
	# -- left boundary, line boundary 
	bcPin_line_left  =  CompiledSubDomain("near(x[0], sideX, TOL) && near(x[1],sideY, TOL)", sideX = x_lo,sideY=0,TOL=10E-5) #y_lo <-- to pin on bottom instead of center
	# -- right boundary, line boundary 
	bcPin_line_right =  CompiledSubDomain("near(x[0], sideX, TOL) && near(x[1],sideY, TOL)", sideX = x_hi,sideY=0,TOL=10E-5) #y_lo <-- to pin on bottom instead of center 
	# -- left boundary, whole side area
	bcDom_xLo = CompiledSubDomain("near(x[0], side) ", side = x_lo)
	# -- right boundary, whole side area
	bcDom_xHi = CompiledSubDomain("near(x[0], side)", side = x_hi)

	# Apply boundary conditions, traction, and body forces
	# --> hypothetically, traction could be applied on the right edge, but this isn't 
	#				implemented in this version of the code 
	boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
	boundary_markers.set_all(0)
	bcDom_xHi.mark(boundary_markers, 1) # Prescribed traction
	ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

	# Define finite element problem
	up = Function(W)
	(u, p) = split(up)
	dup = TrialFunction(W)
	vq = TestFunction(W)

	T  = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary
	integrals_N = [dot(T,u)*ds(1)] # just = 0 here 

	# Problem solver function
	##########################################################################################
	def problem_solve(applied_disp,up,dup,vq, B_input):
		(u, p) = split(up)
		
		B  = Constant((0.0, B_input, 0.0)) 
		
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
		J    = det(F)

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
	def get_disp(u):
		r = x_hi - x_lo; ct = (x_lo + x_hi)/2.0
		pt_list_x = [ ct-r/3.0, ct-r/6.0 , ct, ct+r/6.0, ct+ r/3.0]
		pt_y = 0.0
		pt_z = 0.0

		fiducial_array = np.array([])
		for kk in [0,1,2,3,4]:
			for jj in [0,1,2]:
				fiducial_array = np.append(fiducial_array, u(pt_list_x[kk],pt_y,pt_z)[jj])

		return fiducial_array

	def get_rxn_force(W,f_int,f_ext):
		x_dofs = W.sub(0).sub(0).dofmap().dofs()
		y_dofs = W.sub(0).sub(1).dofmap().dofs()
		z_dofs = W.sub(0).sub(2).dofmap().dofs()
		
		f_ext_known = assemble(f_ext)
		f_ext_unknown = assemble(f_int) - f_ext_known
		dof_coords = W.tabulate_dof_coordinates().reshape((-1, 3))

		x_val_min = np.min(dof_coords[:,0]) + 10E-5; x_val_max = np.max(dof_coords[:,0]) - 10E-5

		y_val_min = np.min(dof_coords[:,1]) + 10E-5; y_val_max = np.max(dof_coords[:,1]) - 10E-5
		
		y_val_min = -1 * 10E-5; y_val_max = 10E-5
		
		x_min = []; x_max = [] 
		for kk in x_dofs:
			if dof_coords[kk,0] < x_val_min and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				x_min.append(kk)
			if dof_coords[kk,0] > x_val_max and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				x_max.append(kk)
		f_sum_left_x = np.sum(f_ext_unknown[x_min])
		f_sum_right_x = np.sum(f_ext_unknown[x_max])		

		y_min = []; y_max = [] 
		for kk in y_dofs:
			if dof_coords[kk,0] < x_val_min and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				y_min.append(kk)
			if dof_coords[kk,0] > x_val_max and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				y_max.append(kk)
		f_sum_left_y = np.sum(f_ext_unknown[y_min])
		f_sum_right_y = np.sum(f_ext_unknown[y_max])		

		z_min = []; z_max = [] 
		for kk in z_dofs:
			if dof_coords[kk,0] < x_val_min and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				z_min.append(kk)
			if dof_coords[kk,0] > x_val_max and dof_coords[kk,1] > y_val_min and dof_coords[kk,1] < y_val_max:
				z_max.append(kk)
		f_sum_left_z = np.sum(f_ext_unknown[z_min])
		f_sum_right_z = np.sum(f_ext_unknown[z_max])		

		print("x_left, x_right rxn force:", f_sum_left_x, f_sum_right_x)
		print("y_left, y_right rxn force:", f_sum_left_y, f_sum_right_y)
		print("z_left, z_right rxn force:", f_sum_left_z, f_sum_right_z)

		return np.array([f_sum_left_x, f_sum_right_x, f_sum_left_y, f_sum_right_y, f_sum_left_z, f_sum_right_z])

	def write_paraview(fname,step_num,u):
		# save displacement
		fname << (u,step_num)

	##########################################################################################
	# actually run the code, call post-processing steps within the loop 
	##########################################################################################
	fname_paraview = File("./iteration%i/disp.pvd"%(num))

	disp_list = np.linspace(exp.ideal_disp, )
	disp_list = np.arange(exp.ideal_disp, -3.4, -0.05)
	step_num = 0
	for kk in range(0,len(disp_list)):
		applied_disp = disp_list[kk]
		print("step number:", step_num)
		# -- run the code 
		up, dup, vq, f_int, f_ext = problem_solve(applied_disp,up,dup,vq,body_force)
		(u,p) = up.split(True)
		# -- post process
		if step_num == 0:
			force_array = get_rxn_force(W,f_int,f_ext)
			u_array = get_disp(u)
		else:
			force_array = np.vstack((force_array, get_rxn_force(W,f_int,f_ext)))
			u_array = np.vstack((u_array, get_disp(u)))

		write_paraview(fname_paraview,step_num,u)
		step_num += 1

	# Save experiment to file
	exp.disp_list = disp_list
	exp.force_array = force_array
	exp.u_array = u_array
	save_experiment('./iteration%i/data.pkl'%num, exp)

