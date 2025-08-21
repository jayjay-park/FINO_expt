#!/usr/bin/env python
"""
Class definition of Laminar-Jet PDE model
written in FEniCS 1.6.0 and 1.7.0-dev, portable to other PDE models
Shiwei Lan @ U of Warwick, 2016
-----------------------------------
The purpose of this script is to obtain geometric quantities, misfit, its gradient and the associated metric (Gauss-Newton) using adjoint methods.
--To run demo:                     python Laminar.py # to compare with the finite difference method
--To initialize problem:     e.g.  laminar=Laminar(args); inflow=laminar.inflow_profile(args)
--To obtain observations:          obs,idx,loc=laminar.get_obs(inflow) # observation values, dof indices, and locations resp.
--To define data misfit class:     misfit=laminar.data_misfit(args)
--To obtain geometric quantities:  nll,dnll,Fv,FI = laminar.get_geom # misfit value, gradient, metric action and metric resp.
                                   which calls soln_fwd, get_grad (soln_adj), get_metact (soln_fwd2,soln_adj2), and get_met resp.
--To save PDE solutions:           laminar.save()
                                   fwd: forward solution; adj: adjoint solution; fwd2: 2nd order forward; adj2: 2nd order adjoint.
--To plot PDE solutions:           laminar.plot()
-----------------------------------
Created March 11, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__credits__ = ["Patrick Farrell","Umberto Villa"]
__license__ = "GPL"
__version__ = "9.0"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com"

# import modules
from dolfin import *
import ufl
# from dolfin_adjoint import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import warnings

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class Laminar:
    def __init__(self, unit=1.0,nx=40,ny=30,nu=1.0e-1,beta=0.5,stokes=False,nugg=1.0e-20,cont=False):
        # 1. Define the Geometry
        # space label
        self.VELOCITY = 0; self.PRESSURE = 1; self.LAGRANGE = 2
        # boundary label
        self.INLET = 1; self.OUTLET = 2; self.BOUNDING = 3

        self.unit=unit; self.Lx = 10*self.unit; self.Ly = 8*self.unit;
        self.nx=nx; self.ny=ny;
        # set FEM
        self.set_FEM()

        # parameter values
        self.nu = Constant(nu)
        self.beta = Constant(beta)
        self.stokes = stokes # indicator of linear problem -- stokes equation
        self.nugg = Constant(nugg)
        self.cont = cont # whether do continuation

        # count PDE solving times
        self.soln_count = np.zeros(6)
        # 0: number of Newton iterations; 1: number of convergences;
        # 2-5: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively

    def set_FEM(self):
        self.mesh = RectangleMesh(Point(0.0, -0.5*self.Ly), Point(self.Lx, 0.5*self.Ly), self.nx, self.ny)

        # boundaries
        bdy_inlet = AutoSubDomain(lambda x, on_boundary: near(x[0], 0.0) and on_boundary)
        self.bdy_outlet = AutoSubDomain(lambda x, on_boundary: near(x[0], self.Lx) and on_boundary)
        bdy_bounding = AutoSubDomain(lambda x, on_boundary: near(abs(x[1]), 0.5*self.Ly) and on_boundary)

        self.boundaries = FacetFunction("size_t", self.mesh, 0)
        self.ds = ds(subdomain_data=self.boundaries)

        bdy_inlet.mark(self.boundaries, self.INLET)
        self.bdy_outlet.mark(self.boundaries, self.OUTLET)
        bdy_bounding.mark(self.boundaries, self.BOUNDING)

        # 2. Define the finite element spaces
        self.V_velocity = VectorFunctionSpace(self.mesh, 'CG', 2)
        self.V_velocity0 = FunctionSpace(self.mesh, 'CG', 2)
        V_pressure = FunctionSpace(self.mesh, 'CG', 1)
        V_lagrange = FunctionSpace(self.mesh, 'CG', 2)

        self.V = MixedFunctionSpace([self.V_velocity, V_pressure, V_lagrange])

        # 3. Define boundary conditions
        # Dirichlet on the inlet (u\cdot n=theta) and bounding (u\cdot n=0)
#         bc_inlet = DirichletBC(V.sub(self.VELOCITY).sub(0), u_inflow, self.boundaries, self.INLET)
        bc_bounding = DirichletBC(self.V.sub(self.VELOCITY).sub(1),Constant(0.0), self.boundaries, self.BOUNDING)
        bc_lagrange = DirichletBC(self.V.sub(self.LAGRANGE), Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS")

        self.ess_bc = [bc_bounding,bc_lagrange]

        # Create adjoint boundary conditions (homogenized forward BCs)
        def homogenize(bc):
            bc_copy = DirichletBC(bc)
            bc_copy.homogenize()
            return bc_copy
        self.adj_bcs = [homogenize(bc) for bc in self.ess_bc]

        # get dof indecies on inlet boundary in V_velocity0
        bc_inlet = DirichletBC(self.V_velocity0, 1, bdy_inlet)
        self.idx_inlet = np.array(bc_inlet.get_boundary_values().keys(),dtype=np.int32)
        # dof coordinates on inlet boundary
        try:
            dof_coordinates = self.V_velocity0.tabulate_dof_coordinates() # post v1.6.0
        except AttributeError:
            print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
            dof_coordinates = self.V_velocity0.dofmap().tabulate_all_coordinates(self.mesh)
        dof_coordinates.resize((self.V_velocity0.dim(), self.mesh.geometry().dim()))
        self.dof_coords_inlet = dof_coordinates[self.idx_inlet,1] # save y coordinate on inlet boundary for the sparsity of following computation
        # move downward to compute in time if it is too large

    # subclass of Expression with varying parameters
    class inflow_profile(Expression): # Karhunen-Loeve expansion of inflow profile # TODO: write cpp code
        def __init__(self,theta,sigma=1.2,alpha=0.1,s=0.6):
            self.theta=theta
            self.sigma=sigma
            self.alpha=alpha
            self.s=s
            self.l=len(self.theta)
        # K-L expansion of theta ~ GP(0,C)
        def eval(self,value,x):
            seq0l = np.arange(self.l,dtype=np.float)
            value[0] = self.sigma*self.theta.dot(pow(self.alpha+(pi*seq0l)**2,-self.s/2)*np.cos(pi*seq0l*x[1]))

    def set_forms(self,inflow,obj=None,ord=[0],opt=3,idx=[]):
        if not any(idx) and opt<=2:
            idx = range(inflow.l)
        if any(s>=0 for s in ord):
            ## forms for forward equation ##
            # 4. Define variational problem
            # functions
            if not(self.cont and hasattr(self, 'states_fwd')):
                self.states_fwd = Function(self.V)
            u, p, l = split(self.states_fwd)
            v, q, m = TestFunctions(self.V)
            # variational forms
            strain = lambda u: sym(grad(u))
            neg = lambda u: (u-abs(u))/2
            n = FacetNormal(self.mesh)
            u_inflow = interpolate(inflow, self.V_velocity0)
            self.F = (
                      self.nu*inner(strain(u),strain(v))*dx + Constant(1.0-self.stokes)*inner(grad(u)*u,v)*dx - p*div(v)*dx
                      + Constant(1.0-self.stokes)*self.beta*neg(inner(u,n))*inner(u,v)*self.ds(self.OUTLET) - div(u)*q*dx
                      - l*inner(v,n)*self.ds(self.INLET)
                      -(inner(u,n)+u_inflow)*m*self.ds(self.INLET)
                      + self.nugg*l*m*dx
                      )
            self.dFdstates = derivative(self.F, self.states_fwd) # Jacobian
            if not(self.cont or self.stokes): # simpler problem for initial guess
                self.F_init = (
                               Constant(.1)*inner(strain(u),strain(v))*dx + inner(grad(u)*u,v)*dx - p*div(v)*dx
                               + self.beta*neg(inner(u,n))*inner(u,v)*self.ds(self.OUTLET) - div(u)*q*dx
                               - l*inner(v,n)*self.ds(self.INLET)
                               -(inner(u,n)+u_inflow)*m*self.ds(self.INLET)
                               + self.nugg*l*m*dx
                              )
        if any(s>=1 for s in ord):
            ## forms for adjoint equation ##
            # Set up the objective functional J
#             u,_,_ = split(self.states_fwd)
#             J_form = obj.form(u)
            # Compute adjoint of forward operator
#             self.dFdstates = derivative(self.F, self.states_fwd)    # linearized forward operator
            args = ufl.algorithms.extract_arguments(self.dFdstates) # arguments for bookkeeping
            self.adj_dFdstates = adjoint(self.dFdstates, reordered_arguments=args) # adjoint linearized forward operator
#             self.dJdstates = derivative(J_form, self.states_fwd, TestFunction(self.V)) # derivative of functional with respect to solution
#             self.dirac_1 = obj.ptsrc(u,1) # dirac_1 cannot be initialized here because it involves evaluation
            ## forms for gradient ##
            self.dFdunknown = derivative(self.F, u_inflow)
            self.adj_dFdunknown = adjoint(self.dFdunknown)

            # obtain compressed dunknown/dtheta, i.e. du_inflow/dtheta
            if not hasattr(self, 'dunknowndtheta_mat'):
#                 try:
#                     dof_coordinates = self.V_velocity0.tabulate_dof_coordinates() # post v1.6.0
#                 except AttributeError:
#                     print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
#                     dof_coordinates = self.V_velocity0.dofmap().tabulate_all_coordinates(self.mesh)
#                 dof_coordinates.resize((self.V_velocity0.dim(), self.mesh.geometry().dim()))
                seq = np.arange(inflow.l,dtype=np.float);
                if opt==1:# or (opt==2 and any(s>1 for s in ord)):
                    seq = seq[idx]
                seq = seq[:,None]
#                 self.dunknowndtheta_mat = inflow.sigma*pow(inflow.alpha+(pi*seq)**2,-inflow.s/2)*np.cos(pi*seq*dof_coordinates[None,self.idx_inlet,1]) # TODO: move to __set_FEM__?
                self.dunknowndtheta_mat = inflow.sigma*pow(inflow.alpha+(pi*seq)**2,-inflow.s/2)*np.cos(pi*seq*self.dof_coords_inlet[None,])

        if 2 in ord:
#             ## forms for 2nd adjoint equation ##
# #             self.d2Jdstates = derivative(self.dJdstates, self.states_fwd) # 2nd order derivative of functional with respect to solution
#             self.dirac_2 = obj.ptsrc(ord=2) # dirac_1 cannot be initialized here because it is independent of u
            # create sparse matrix with scipy
            if not hasattr(self, 'dunknowndtheta_sps'):
                dunknowndtheta2sps = self.dunknowndtheta_mat.T
                if opt==2:
                    dunknowndtheta2sps = dunknowndtheta2sps[:,idx]
                dunknowndtheta_sps = sps.lil_matrix((self.V_velocity0.dim(),dunknowndtheta2sps.shape[1]))
                dunknowndtheta_sps[self.idx_inlet,:] =  dunknowndtheta2sps
                self.dunknowndtheta_sps = dunknowndtheta_sps.tocsr()
    def init_guess(self,rel_tol=1e-1):
        # solve a simpler problem for initial guess
        problem = NonlinearVariationalProblem(self.F_init, self.states_fwd, self.ess_bc, J=derivative(self.F_init, self.states_fwd))
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']["relative_tolerance"] = rel_tol
        solver.parameters['newton_solver']['error_on_nonconvergence'] = False
        newton_iter,congv_ind = solver.solve()
        self.soln_count[:3] += [newton_iter,congv_ind,1]

    def soln_fwd(self,rel_tol=1e-5):
        # initialize states
#         den = Expression("A+B*x[0]", A=self.unit, B=Constant(0.06))
#         states_init = Expression( ("A/C*(0.5 + 0.5*tanh( 10*(-fabs(x[1]) + 0.5*C)/C ) )", "0.", "0.", "0."), A=self.unit, C=den)
        states_init = Constant((0.0, 0.0, 0.0, 1.0))
        self.states_fwd.interpolate(states_init)
        # initial guess for nonlinear problem
        if not(self.cont or self.stokes):
            self.init_guess()

        # 5. Solve using Newton solver
        problem = NonlinearVariationalProblem(self.F, self.states_fwd, self.ess_bc, J=self.dFdstates)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']["relative_tolerance"] = rel_tol
        solver.parameters['newton_solver']['error_on_nonconvergence'] = True
        newton_iter,congv_ind = solver.solve()
        self.soln_count[:3] += [newton_iter,congv_ind,1]
        u_fwd, p_fwd, l_fwd = split(self.states_fwd)
        return u_fwd, p_fwd, l_fwd

    def get_obs(self,inflow):
        # obtain the solution of u in a finer mesh
        self.nx*=2; self.ny*=2
        self.set_FEM(); self.set_forms(inflow)
        _,_,_=self.soln_fwd()
        u,_,_=self.states_fwd.split(True)
        u_vec = u.vector().array()
        # choose locations on the outlet boundary
        bmesh = BoundaryMesh(self.mesh, "exterior", True)
        bmesh_coordinates = bmesh.coordinates()
        ind_on_outlet = [self.bdy_outlet.inside(Point(p),True) for p in bmesh_coordinates]
        loc = bmesh_coordinates[np.where(ind_on_outlet)[0],]
        idx,loc,_ = self.in_dof(loc,tol=1e-6) # dof index in V_velocity
        # obtain observations
        obs = u_vec[idx]
#         print(idx)
        # reset to the original coarser mesh
        self.nx/=2; self.ny/=2
        self.set_FEM();
        self.states_fwd = interpolate(self.states_fwd,self.V)
#         del self.states_fwd
        # update indices, locations and observations
        bmesh = BoundaryMesh(self.mesh, "exterior", True)
        bmesh_coordinates = bmesh.coordinates()
        ind_on_outlet = [self.bdy_outlet.inside(Point(p),True) for p in bmesh_coordinates]
        loc = bmesh_coordinates[np.where(ind_on_outlet)[0],]
        idx,loc,rel_idx = self.in_dof(loc,tol=1e-6)
        obs = obs[rel_idx]
#         print(idx)
        return obs,idx,loc

    def in_dof(self,points,V=None,tol=2*DOLFIN_EPS): # generic function to determine whether points are nodes where dofs are defined and output those dofs
        # obtain coordinates of dofs
        d = self.mesh.geometry().dim()
        if V is None:
            n = self.V_velocity.dim() # 2 dofs associated to each point, the smaller one corresponds to u[0]
            try:
                dof_coordinates = self.V_velocity.tabulate_dof_coordinates() # post v1.6.0
            except AttributeError:
                print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
                dof_coordinates = self.V_velocity.dofmap().tabulate_all_coordinates(self.mesh)
            dof_coordinates.resize((n, d))
        else:
            n = V.dim() # V should NOT be mixed function space! Unless you know what you are doing...
            if V.num_sub_spaces()>1:
                print('Warning: Multiple dofs associated with each point, unreliable outputs!')
            try:
                dof_coordinates = V.tabulate_dof_coordinates() # post v1.6.0
            except AttributeError:
                print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
                dof_coordinates = V.dofmap().tabulate_all_coordinates(self.mesh)
            dof_coordinates.resize((n, d))
        # check whether those points are close to nodes where dofs are defined
        pdist_pts2dofs = np.einsum('ijk->ij',(points[:,None,:]-dof_coordinates[None,:,:])**2)
        idx_in_dof = np.argmin(pdist_pts2dofs,axis=1)
        rel_idx_in = np.where(np.einsum('ii->i',pdist_pts2dofs[:,idx_in_dof])<tol**2)[0] # index relative to points
        idx_in_dof = idx_in_dof[rel_idx_in]
        loc_in_dof = points[rel_idx_in,]
        return idx_in_dof,loc_in_dof,rel_idx_in

    class _data_misfit(object):
        def __init__(self,out_obj,obs,prec,idx=None,loc=None):
            self.out_obj=out_obj
            self.obs = obs
            self.prec = prec
            self.idx = idx
            self.loc = loc
#             # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#             # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#             pts_domain = VertexFunction("size_t", self.out_obj.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #             pts_nbhd = AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#             pts_nbhd = AutoSubDomain(lambda x: any([Point(x).distance(Point(p))<2*DOLFIN_EPS for p in self.loc]))
#             pts_nbhd.mark(pts_domain, 1)
#             self.dpm = dP(subdomain_data=pts_domain)
            # find global dof of observations
            idx_dirac_local,_,self.idx_dirac_rel2Vv = self.out_obj.in_dof(self.loc, self.out_obj.V_velocity0) # idx_dirac_rel2Vv: indices relative to V_velocity
            sub_dofs = self.out_obj.V.sub(self.out_obj.VELOCITY).sub(0).dofmap().dofs() # dof map: V_velocity0 --> V
            self.idx_dirac_global = sub_dofs[idx_dirac_local]
        def extr_sol_vec(self,u):
            # u_vec: solution vector on observation locations
            if type(u) is ufl.tensors.ListTensor:
                u_vec = [u[0](list(x)) for x in self.loc]
            elif type(u) is dolfin.functions.function.Function:
                if self.idx is not None:
                    u_vec = u.vector()[self.idx]
                elif self.loc is not None:
                    u_vec = [u(list(x))[0] for x in self.loc]
            elif type(u) is dolfin.cpp.la.GenericVector or np.ndarray:
                u_vec = u[self.idx]
            else:
                raise Exception('Check the type of u! Either the indeces or the locations of observations are needed!')
            return np.array(u_vec)
        def eval(self,u):
            u_vec = self.extr_sol_vec(u)
            diff = u_vec-self.obs
            val = 0.5*self.prec*diff.dot(diff)
            return val
#         def func(self,u):
#             if type(u) is not ufl.tensors.ListTensor:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             f_ind = Function(self.out_obj.V_velocity)
# #             f_ind.vector()[:] = 0
#             f_ind.vector()[self.idx] = 1
#             u_obs = Function(self.out_obj.V_velocity)
#             u_obs.vector()[self.idx] = self.obs
#             fun = 0.5*self.prec*(inner(u,f_ind)-u_obs[0])**2
#             return fun
#         def form(self,u):
#             if type(u) is not ufl.tensors.ListTensor:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#             # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#             pts_domain = VertexFunction("size_t", self.out_obj.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #             pts_nbhd = AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#             pts_nbhd = AutoSubDomain(lambda x: any([Point(x).distance(Point(p))<2*DOLFIN_EPS for p in self.loc]))
#             pts_nbhd.mark(pts_domain, 1)
#             self.dpm = dP(subdomain_data=pts_domain)
#             # u_obs function with observation values supported on observation locations
#             u_obs = Function(self.out_obj.V_velocity)
#             u_obs.vector()[self.idx] = self.obs
#             fom = 0.5*self.prec*(u[0]-u_obs[0])**2*self.dpm(1)
#             return fom
        def ptsrc(self,u,ord=1):
            u_vec = self.extr_sol_vec(u)
            # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
            dfun_vec = u_vec
            if ord==1:
                dfun_vec -= self.obs
            dfun_vec *= self.prec
            dirac = [PointSource(self.out_obj.V.sub(self.out_obj.VELOCITY).sub(0),Point(p),f) for (p,f) in zip(self.loc,dfun_vec)]
            return dirac
#         def ptsrc1(self,u,ord=1):
#             if type(u) is not ufl.tensors.ListTensor:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
#             dfun = u
#             if ord==1:
#                 u_obs = Function(self.out_obj.V_velocity)
#                 u_obs.vector()[self.idx] = self.obs
#                 dfun -= u_obs
#             dfun *= self.prec
#             dirac = [PointSource(self.out_obj.V.sub(self.out_obj.VELOCITY).sub(0),Point(p),dfun[0](list(p))) for p in self.loc]
#             return dirac
        def dirac(self,u,ord=1):
            u_vec = self.extr_sol_vec(u)
            dfun_vec = u_vec[self.idx_dirac_rel2Vv]
            if ord==1:
                dfun_vec -= self.obs
            dfun_vec *= self.prec
            return dfun_vec,self.idx_dirac_global

    def data_misfit(self,obs,prec,idx=None,loc=None):
        return self._data_misfit(self,obs,prec,idx,loc)

    def soln_adj(self,obj):
        self.states_adj = Function(self.V) # adjoint states
        # Solve adjoint PDE < adj_dFdstates, states_adj > = dJdstates
#         solve(self.adj_dFdstates == self.dJdstates , self.states_adj, self.adj_bcs)
#         A,b = assemble_system(self.adj_dFdstates, self.dJdstates, self.adj_bcs)
#         solve(A, self.states_adj.vector(), b)
#         self.adj_dFdstates_assemb = PETScMatrix(); dJdstates_assemb = PETScVector()
#         assemble_system(self.adj_dFdstates, self.dJdstates, self.adj_bcs, A_tensor=self.adj_dFdstates_assemb, b_tensor=dJdstates_assemb)
#         solve(self.adj_dFdstates_assemb, self.states_adj.vector(), dJdstates_assemb)
        # error: assemble (solve) point integral (J) has supported underlying FunctionSpace more than CG1
        # have to use PointSource? Yuk!

        self.adj_dFdstates_assemb = PETScMatrix();
        assemble(self.adj_dFdstates, tensor=self.adj_dFdstates_assemb)

        u_fwd,_,_ = split(self.states_fwd)
        if not has_petsc4py():
            warnings.warn('Configure dolfin with petsc4py to run faster!')
            self.dirac_1 = obj.ptsrc(u_fwd,ord=1)
            rhs_adj = Vector(mpi_comm_world(),self.V.dim())
            [delta.apply(rhs_adj) for delta in self.dirac_1]
        else:
            rhs_adj = PETScVector(mpi_comm_world(),self.V.dim())
            val_dirac_1,idx_dirac_1 = obj.dirac(u_fwd,ord=1)
            rhs_adj.vec()[idx_dirac_1] = val_dirac_1
#             np.allclose(rhs_adj.array(),rhs_adj1.vec())

        [bc.apply(self.adj_dFdstates_assemb,rhs_adj) for bc in self.adj_bcs]

        solve(self.adj_dFdstates_assemb, self.states_adj.vector(), rhs_adj)
        self.soln_count[3] += 1
        u_adj, p_adj, l_adj = split(self.states_adj)
        return u_adj, p_adj, l_adj

    def get_grad(self,obj):
        # solve adjoint equations
        _,_,_ = self.soln_adj(obj)
        # compute the gradient of dJ/dunknown = - <states_adj, adj_dFdunknown> + (dJdunknown=0)
        g_unknown_form = -action(self.adj_dFdunknown,self.states_adj)
        g_unknown_vec = assemble(g_unknown_form)
#         g_unknown = Function(self.V_velocity0)
#         g_unknown.vector()[:] = g_unknown_vec
#         plot(g_unknown, title='gradient', rescale=True)
#         interactive()
        # the desired gradient dJ/dtheta = dunknown/dtheta * dJ/dunknown
        g_theta = self.dunknowndtheta_mat.dot(g_unknown_vec[self.idx_inlet])

        return g_theta

    def soln_fwd2(self,u_actedon):
        if type(u_actedon) is np.ndarray:
            u = Function(self.V_velocity0)
#             u.vector()[self.idx_inlet] = self.dunknowndtheta_mat.T.dot(u_actedon)
            dunknowndtheta_spsT = self.dunknowndtheta_sps[self.idx_inlet,]
            u.vector()[self.idx_inlet] = dunknowndtheta_spsT.toarray().dot(u_actedon)
            u_actedon = u

        self.states_fwd2 = Function(self.V) # 2nd forward states
        # Solve 2nd forward PDE < dFdstates, states_fwd2 > = < dFdunknown, u_actedon >
#         solve(self.dFdstates == action(self.dFdunknown, u_actedon), self.states_fwd2, self.adj_bcs) # ToDo: check the boundary for fwd2
#         A,b = assemble_system(self.dFdstates, action(self.dFdunknown, u_actedon), self.adj_bcs)
#         solve(A, self.states_fwd2.vector(), b)

#         if not hasattr(self, 'dFdstates_assemb'):
#             self.dFdstates_assemb = PETScMatrix()
#             assemble(self.dFdstates, tensor=self.dFdstates_assemb)
#             [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]

        rhs_fwd2 = PETScVector()
#         assemble(action(self.dFdunknown, u_actedon), tensor=rhs_fwd2)
        self.dFdunknown_assemb.mult(u_actedon.vector(),rhs_fwd2)

        [bc.apply(rhs_fwd2) for bc in self.adj_bcs]

        solve(self.dFdstates_assemb, self.states_fwd2.vector(), rhs_fwd2)
        self.soln_count[4] += 1
        u_fwd2, p_fwd2, l_fwd2 = split(self.states_fwd2)
        return u_fwd2, p_fwd2, l_fwd2

    def soln_adj2(self,obj):
        self.states_adj2 = Function(self.V) # 2nd forward states
        # Solve 2nd adjoint PDE < adj_dFdstates, states_adj2 > = < d2Jdstates, states_fwd2 >
#         solve(self.adj_dFdstates == action(self.d2Jdstates, self.states_fwd2), self.states_adj2, self.adj_bcs)
#         A,b = assemble_system(self.adj_dFdstates, action(self.d2Jdstates, self.states_fwd2), self.adj_bcs)
#         solve(A, self.states_adj2.vector(), b)

#         rhs_adj2 = PETScVector()
#         assemble(action(self.d2Jdstates, self.states_fwd2), tensor=rhs_adj2)

        u_fwd2,_,_ = split(self.states_fwd2)
        if not has_petsc4py():
            warnings.warn('Configure dolfin with petsc4py to run faster!')
            self.dirac_2 = obj.ptsrc(u_fwd2,ord=2)
            rhs_adj2 = Vector(mpi_comm_world(),self.V.dim())
            [delta.apply(rhs_adj2) for delta in self.dirac_2]
        else:
            rhs_adj2 = PETScVector(mpi_comm_world(),self.V.dim())
            val_dirac_2,idx_dirac_2 = obj.dirac(u_fwd2,ord=2)
            rhs_adj2.vec()[idx_dirac_2] = val_dirac_2
#             np.allclose(rhs_adj2.array(),rhs_adj12.vec())

        [bc.apply(rhs_adj2) for bc in self.adj_bcs]

        solve(self.adj_dFdstates_assemb, self.states_adj2.vector(), rhs_adj2)
        self.soln_count[5] += 1
        u_adj2, p_adj2, l_adj2 = split(self.states_adj2)
        return u_adj2, p_adj2, l_adj2

    def get_metact(self,obj,u_actedon):
        # solve 2nd forward/adjoint equations
        _,_,_ = self.soln_fwd2(u_actedon)
        _,_,_ = self.soln_adj2(obj)
        # compute the metric action on u_actedon of d2J/dunknown = < adj_dFdunknown, states_adj2 >
#         Ma_unknown_form = action(self.adj_dFdunknown,self.states_adj2)
#         Ma_unknown_vec = assemble(Ma_unknown_form)
        Ma_unknown_vec = PETScVector()
        self.adj_dFdunknown_assemb.mult(self.states_adj2.vector(),Ma_unknown_vec)
        # the desired metric action d2J/dtheta = dunknown/dtheta * d2J/dunknown
#         Ma_theta = self.dunknowndtheta_mat.dot(Ma_unknown_vec[self.idx_inlet])
        dunknowndtheta_spsT = self.dunknowndtheta_sps[self.idx_inlet,]
        Ma_theta = dunknowndtheta_spsT.toarray().T.dot(Ma_unknown_vec[self.idx_inlet])

        return Ma_theta

    def get_met(self,obj):
        """
        Get metric by solving adjoints with multiple RHS' simultaneously-- about 10 times faster than using get_metact on unit vectors
        """
#         import scipy.sparse as sps
#         import scipy.sparse.linalg

        #-- All 2nd forward states --#
        # Solve 2nd forward PDE < dFdstates, states_fwd2s > = < dFdunknown, I >
        # Convert PETScMatrix to csr_matrix
        petscmat2csr = lambda matrix: sps.csr_matrix(tuple(as_backend_type(matrix).mat().getValuesCSR()[::-1]),
                                                     shape=(matrix.size(0),matrix.size(1)))

        dFdunknown_sps = petscmat2csr(self.dFdunknown_assemb)
        rhs_fwd2s_sps = dFdunknown_sps.dot(self.dunknowndtheta_sps) # multiple rhs_fwd2s

        ### apply homogenized boundary condition to the right hand side vectors using Scipy sparse matrix ###

        # Convert csr_matrix to pestc_mat
#         csr2petscmat = lambda matrix: PETSc.Mat().createAIJ(size=matrix.shape,csr=(matrix.indptr,matrix.indices,matrix.data))
#         rhs_fwd2s_petsc = PETScMatrix(csr2petscmat(rhs_fwd2s_sps))
#         [bc.apply(rhs_fwd2s_petsc) for bc in self.adj_bcs]
#         [bc.zero(rhs_fwd2s_petsc) for bc in self.adj_bcs] # neither works
        """
        Some function to efficiently zero-out certain rows in scipy sparse matrix (csr)
        http://stackoverflow.com/questions/19784868/what-is-most-efficient-way-of-setting-row-to-zeros-for-a-sparse-scipy-matrix
        """
#         def csr_row_set_nz_to_val(csr, row, value=0):
#             """Set all nonzero elements (elements currently in the sparsity pattern)
#             to the given value. Useful to set to 0 mostly.
#             """
#             if not isinstance(csr, scipy.sparse.csr_matrix):
#                 raise ValueError('Matrix given must be of CSR format.')
#             csr.data[csr.indptr[row]:csr.indptr[row+1]] = value
#
#         def csr_rows_set_nz_to_val(csr, rows, value=0):
#             for row in rows:
#                 csr_row_set_nz_to_val(csr, row)
#             if value == 0:
#                 csr.eliminate_zeros()

        def csr_zero_rows(csr, rows_to_zero): # i like this one better
            rows, cols = csr.shape
            mask = np.ones((rows,), dtype=np.bool)
            mask[rows_to_zero] = False
            nnz_per_row = np.diff(csr.indptr)

            mask = np.repeat(mask, nnz_per_row)
            nnz_per_row[rows_to_zero] = 0
            csr.data = csr.data[mask]
            csr.indices = csr.indices[mask]
            csr.indptr[1:] = np.cumsum(nnz_per_row)
        """------------------end--of--function----------------"""
        # homogenized Dirichlet boundary condition
        for bc in self.adj_bcs:
            binds = bc.get_boundary_values().keys()
            csr_zero_rows(rhs_fwd2s_sps,binds)

        states_fwd2s_sps = sps.linalg.spsolve(petscmat2csr(self.dFdstates_assemb),rhs_fwd2s_sps) # csc? # all 2nd forward states simultaneously
        self.soln_count[4] += states_fwd2s_sps.shape[1]

        #-- All 2nd adjoint states --#
        # Solve 2nd adjoint PDE < adj_dFdstates, states_adj2s > = < d2Jdstates, states_fwd2s >
#         rhs_adj2s_sps = sps.csr_matrix(states_fwd2s_sps.shape)
#         rhs_adj2s_sps[obj.idx_dirac,] = obj.prec*states_fwd2s_sps[obj.idx_dirac,]
        def csr_keep_rows(csr, rows_to_keep): # modified from above
            rows, cols = csr.shape
            labl = np.zeros((rows,), dtype=np.bool)
            labl[rows_to_keep] = True
            nnz_per_row = np.diff(csr.indptr)

            labl = np.repeat(labl, nnz_per_row)
            nnz_per_row_kept = np.zeros_like(nnz_per_row)
            nnz_per_row_kept[rows_to_keep] = nnz_per_row[rows_to_keep]
            csr.data = csr.data[labl]
            csr.indices = csr.indices[labl]
            csr.indptr[1:] = np.cumsum(nnz_per_row_kept)

        rhs_adj2s_sps = obj.prec*states_fwd2s_sps # multiple rhs_adj2s
        csr_keep_rows(rhs_adj2s_sps,obj.idx_dirac_global) # using global dof index to avoid Point Source
#         csr_zero_rows(rhs_adj2s_sps,np.setdiff1d(range(states_fwd2s_sps.shape[0]),obj.idx_dirac))

        # homogenized Dirichlet boundary condition
        for bc in self.adj_bcs:
            binds = bc.get_boundary_values().keys()
            csr_zero_rows(rhs_adj2s_sps,binds)

        states_adj2s_sps = sps.linalg.spsolve(petscmat2csr(self.adj_dFdstates_assemb),rhs_adj2s_sps) # csc? # all 2nd adjoint states simultaneously
        self.soln_count[5] += states_adj2s_sps.shape[1]

        # compute the metric action on u_actedon of d2J/dunknown = < adj_dFdunknown, states_adj2 >
        M_unknown_sps = petscmat2csr(self.adj_dFdunknown_assemb).dot(states_adj2s_sps)
        # the desired metric action d2J/dtheta = dunknown/dtheta * d2J/dunknown
#         M_unknown_mat = M_unknown_sps[self.idx_inlet,]
#         M_theta = self.dunknowndtheta_mat.dot(M_unknown_mat.toarray())
        M_theta = self.dunknowndtheta_sps.T.dot(M_unknown_sps)
        M_theta = M_theta.toarray()

        return M_theta

    def get_geom(self,inflow,obj=None,ord=[0],opt=3,idx=[],log_level=ERROR):
        val=None; grad=None; metact=None; met=None;
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        set_log_level(log_level)
        if not any(idx) and opt<=2:
            idx = range(inflow.l) # idx: indices of parameters' components left
        if opt==0:
            inflow.theta = inflow.theta[idx] # opt: the derivative order where truncation starts: 2 only for metric; 1 for both gradient and metric; 0 for all geometrics
            inflow.l = len(idx)
        self.set_forms(inflow=inflow,obj=obj,ord=ord,opt=opt,idx=idx)
#         import time
#         start=time.time()
        if any(s>=0 for s in ord):
            u,_,_ = self.soln_fwd()
            val = obj.eval(u)
#         end=time.time()
#         print('Time for obtaining misfit value is %.10f' % (end-start))
#         start=time.time()
        if any(s>=1 for s in ord):
            grad = self.get_grad(obj)
#         end=time.time()
#         print('Time for obtaining the gradient is %.10f' % (end-start))
#         start=time.time()
        if any(s>1 for s in ord):
            # do some assembling here to avoid repetition
            # for fwd2:
            self.dFdstates_assemb = PETScMatrix()
            assemble(self.dFdstates, tensor=self.dFdstates_assemb)
            [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]
            self.dFdunknown_assemb = PETScMatrix()
            assemble(self.dFdunknown, tensor=self.dFdunknown_assemb)
            # for metact:
            self.adj_dFdunknown_assemb = PETScMatrix()
            assemble(self.adj_dFdunknown, tensor=self.adj_dFdunknown_assemb)
        if 1.5 in ord:
            metact = lambda v: self.get_metact(obj,v)
        if 2 in ord:
            if has_petsc4py():
#                 import time
#                 start=time.time()
                met = self.get_met(obj)
#                 end=time.time()
#                 print('k! times is %.10f' % (end-start))
            else:
                warnings.warn('Configure dolfin with petsc4py to run faster!')
                metact = lambda v: self.get_metact(obj,v)
#                 start=time.time()
                met = np.array([metact(e) for e in np.eye(self.dunknowndtheta_sps.shape[1])])
#                 end=time.time()
#                 print('o! times is %.10f' % (end-start))
#         end=time.time()
#         print('Time for obtaining the metric is %.10f' % (end-start))
        return val,grad,metact,met

    def save(self,sep=False):
        # title settings
        self.titles = ['Velocity','Pressure','Lagrange Multiplier']
        self.sols = ['fwd','adj','fwd2','adj2']
        self.sub_titles = ['forward','adjoint','2nd forward','2nd adjoint']
        import os
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        else:
            for j in range(len(self.sols)):
                # get solution
                sol_name = '_'.join(['states',self.sols[j]])
                try:
                    soln = getattr(self,sol_name)
                except AttributeError:
                    print(self.sub_titles[j]+'solution not found!')
                    pass
                else:
                    if not sep:
                        File(os.path.join(self.savepath,sol_name+'.xml'))<<soln
                    else:
                        soln = soln.split(True)
                        for i in range(len(self.titles)-1):
                            File(os.path.join(self.savepath,'_'.join([self.titles[i],self.sols[j]])+'.pvd'))<<soln[i]

    def plot_vtk(self,SAVE=False):
        for i in range(len(self.titles)-1):
            for j in range(len(self.sols)):
                # get solution
                try:
                    soln = getattr(self,'_'.join(['states',self.sols[j]]))
                except AttributeError:
                    print(self.sub_titles[j]+'solution not found!')
                    pass
                else:
                    soln = soln.split(True)
                    fig=plot(soln[i],title=self.sub_titles[j]+' '+self.titles[i],rescale=True)
                    if SAVE:
                        import os
                        if not hasattr(self, 'savepath'):
                            cwd=os.getcwd()
                            self.savepath=os.path.join(cwd,'result')
                            if not os.path.exists(self.savepath):
                                print('Save path does not exist; created one.')
                                os.makedirs(self.savepath)
                        fig.write_png(os.path.join(self.savepath,'_'.join([self.titles[i],self.sols[j]])+'.png'))

    def plot_mpl(self,SAVE=False):
        import matplotlib.pyplot as plt
        try:
            col_bar_supp = True
            parameters["plotting_backend"]="matplotlib"
        except KeyError:
            col_bar_supp = False #no colorbar support on older version
            print('Warning: plot has not been overloaded with matplotlib before version 1.7.0-dev!')
            """
            generic function plotting solution over 2D mesh
            by Chris Richardson @ https://bitbucket.org/fenics-project/dolfin/issues/455/add-ipython-compatible-matplotlib-plotting
            already incorporated in version 1.7.0
            """
            import matplotlib.tri as tri
            def mesh2triang(mesh):
                xy = mesh.coordinates()
                return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

            def mplot_cellfunction(cellfn):
                C = cellfn.array()
                tri = mesh2triang(cellfn.mesh())
                return plt.tripcolor(tri, facecolors=C)

            def mplot_function(f):
                mesh = f.function_space().mesh()
                if (mesh.geometry().dim() != 2):
                    raise AttributeError('Mesh must be 2D')
                # DG0 cellwise function
                if f.vector().size() == mesh.num_cells():
                    C = f.vector().array()
                    return plt.tripcolor(mesh2triang(mesh), C)
                # Scalar function, interpolated to vertices
                elif f.value_rank() == 0:
                    C = f.compute_vertex_values(mesh)
                    return plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
                # Vector function, interpolated to vertices
                elif f.value_rank() == 1:
                    w0 = f.compute_vertex_values(mesh)
                    if (len(w0) != 2*mesh.num_vertices()):
                        raise AttributeError('Vector field must be 2D')
                    X = mesh.coordinates()[:, 0]
                    Y = mesh.coordinates()[:, 1]
                    U = w0[:mesh.num_vertices()]
                    V = w0[mesh.num_vertices():]
                    return plt.quiver(X,Y,U,V)

            # Plot a generic dolfin object (if supported)
            global plot
            def plot(obj):
#                 plt.gca().set_aspect('equal')
                if isinstance(obj, Function):
                    return mplot_function(obj)
                elif isinstance(obj, CellFunctionSizet):
                    return mplot_cellfunction(obj)
                elif isinstance(obj, CellFunctionDouble):
                    return mplot_cellfunction(obj)
                elif isinstance(obj, CellFunctionInt):
                    return mplot_cellfunction(obj)
                elif isinstance(obj, Mesh):
                    if (obj.geometry().dim() != 2):
                        raise AttributeError('Mesh must be 2D')
                    return plt.triplot(mesh2triang(obj), color='#808080')
                else:
                    raise AttributeError('Failed to plot %s'%type(obj))
            '''-----------end--of--plotting--function--by--Chris-Richardson----------------'''
        # codes for plotting solutions
        import matplotlib as mp
        for i in range(len(self.titles)):
            fig,axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,num=i,figsize=(10,6))
            j = 0
            for ax in axes.flat:
                # get solution
                try:
                    soln = getattr(self,'_'.join(['states',self.sols[j]]))
                except AttributeError:
                    print(self.sub_titles[j]+'solution not found!')
                    pass
                else:
                    soln = soln.split(True)
                    plt.axes(ax)
                    sub_fig = plot(soln[i])
                    plt.axis([0.0, self.Lx, -0.5*self.Ly, 0.5*self.Ly])
                    ax.set_title(self.sub_titles[j])
                j += 1
            if col_bar_supp:
                cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
                plt.colorbar(sub_fig, cax=cax, **kw)
            # set common titles
            fig.suptitle(self.titles[i])
            if SAVE:
                import os
                if not hasattr(self, 'savepath'):
                    cwd=os.getcwd()
                    self.savepath=os.path.join(cwd,'result')
                    if not os.path.exists(self.savepath):
                        print('Save path does not exist; created one.')
                        os.makedirs(self.savepath)
                plt.savefig(os.path.join(self.savepath,self.titles[i]+'.png'),bbox_inches='tight')

    def plot(self,backend='matplotlib',SAVE=False):
#         parameters["plotting_backend"]=backend
        # title settings
        if not hasattr(self, 'titles'):
            self.titles = ['Velocity','Pressure','Lagrange Multiplier']
        if not hasattr(self, 'sols'):
            self.sols = ['fwd','adj','fwd2','adj2']
        if not hasattr(self, 'sub_titles'):
            self.sub_titles = ['forward','adjoint','2nd forward','2nd adjoint']
        if backend is 'matplotlib':
            import matplotlib.pyplot as plt
            self.plot_mpl(SAVE=SAVE)
            plt.show()
        elif backend is 'vtk':
            self.plot_vtk(SAVE=SAVE)
            interactive()
        else:
            raise Exception(backend+'not found!')

    def test(self,dim_theta=10,sigma=1.2,alpha=0.1,s=0.6,var_obs=1e-2,chk_fd=False,h=1e-4):
        # generate theta
        theta=np.random.randn(dim_theta)
        inflow=self.inflow_profile(theta=theta,sigma=sigma,alpha=alpha,s=s)
        # obtain observations
        print('Obtaining observations...')
        obs,idx,loc=self.get_obs(inflow)
        num_obs=len(idx)
        print('%d observations have been obtained!' % num_obs)
        # add some noise
        obs += np.sqrt(var_obs)*np.random.randn(num_obs)
        # define data misfit class
        print('\nDefining data-misfit...')
        # smaller problem to reduce waiting time
#         red_num_obs = np.random.randint(1,np.ceil(num_obs/2))
#         obs=obs[:red_num_obs]; idx=idx[:red_num_obs]; loc=loc[:red_num_obs,]
        obs=obs[::3]; idx=idx[::3]; loc=loc[::3,]
        red_num_obs = len(idx)
        print('Reduced to %d observations to save waiting time for results.' % red_num_obs)
        misfit=self.data_misfit(obs,1./var_obs,idx,loc)

        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities with Adjoint method...')
        start = time.time()
        nll,dnll,Fv,FI = self.get_geom(inflow,misfit,[0,1,1.5,2])
        if dnll is not None:
            print('gradient:')
            print(dnll)
        v = np.random.randn(inflow.l)
        if Fv is not None:
            Ma = Fv(v)
            print('metric action on a random vector:')
            print(Ma)
        if FI is not None:
            print('metric:')
            print(FI)
        end = time.time()
        print('Time used is %.4f' % (end-start))

        # save solutions to file
#         self.save()
        # plot solutions
        self.plot()

        if chk_fd:
            # check with finite difference
            print('\n\nTesting against Finite Difference method...')
            start = time.time()
            theta1 = theta.copy(True);
            ## gradient
            print('\nFirst gradient:')
            dnll_fd = np.zeros_like(dnll)
            for i in range(len(theta)):
                theta1[i]+=h; inflow.theta=theta1 # update theta
                nll_p,_,_,_ = self.get_geom(inflow,misfit)
                theta1[i]-=2*h; inflow.theta=theta1
                nll_m,_,_,_ = self.get_geom(inflow,misfit)
                dnll_fd[i] = (nll_p-nll_m)/(2*h)
                theta1[i]+=h;
            print('gradient:')
            print(dnll_fd)
            diff_grad = dnll_fd-dnll
            print('Difference in gradient between adjoint and finite difference: %.10f (inf-norm) and %.10f (2-norm)' % (np.linalg.norm(diff_grad,np.inf),np.linalg.norm(diff_grad)))

            ## metric-action
            print('\nThen Metric-action:')
            Ma_fd = np.zeros_like(Ma)
            # obtain sensitivities
            for n in range(len(idx)):
                misfit_n=self.data_misfit(obs[n],1./var_obs,idx[n],loc[None,n,])
                dudtheta=np.zeros_like(theta)
                for i in range(len(theta)):
                    theta1[i]+=h; inflow.theta=theta1
                    self.set_forms(inflow)
                    u_p,_,_ = self.soln_fwd()
                    u_p_vec = misfit_n.extr_sol_vec(u_p)
                    theta1[i]-=2*h; inflow.theta=theta1
                    self.set_forms(inflow)
                    u_m,_,_ = self.soln_fwd()
                    u_m_vec = misfit_n.extr_sol_vec(u_m)
                    dudtheta[i]=(u_p_vec-u_m_vec)/(2*h)
                    theta1[i]+=h;
                Ma_fd += dudtheta*(dudtheta.dot(v))
            Ma_fd *= misfit.prec
            print('metric action on a random vector:')
            print(Ma_fd)
            diff_Ma = Ma_fd-Ma
            print('Difference in metric-action between adjoint and finite difference: %.10f (inf-norm) and %.10f (2-norm)' % (np.linalg.norm(diff_Ma,np.inf),np.linalg.norm(diff_Ma)))
            end = time.time()
            print('Time used is %.4f' % (end-start))

if __name__ == '__main__':
    laminar = Laminar(unit=.1,nx=40,ny=40,nu=4.0e-2,stokes=True)
    laminar.test(dim_theta=10,sigma=.1,alpha=1,chk_fd=True)

