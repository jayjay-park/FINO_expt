using LinearAlgebra
using SparseArrays
using ChainRulesCore: rrule, unthunk, NoTangent, @not_implemented
using Jutul: Jutul, simulate, setup_parameter_optimization, optimization_config, vectorize_variables, SimulationModel, MultiModel, submodels, submodels_symbols, get_primary_variables, Simulator, linear_operator
using JutulDarcy
using JutulDarcyRules
using Flux: gradient
using Test
using Random
using Printf
using JutulDarcy: Pressure
using LinearOperators
using IterativeSolvers

########################################################################
# 1. Simulator Update and Forward–Mode JVP Routine
########################################################################

# Update the simulator state and reassemble its sensitivity system.
function update_simulator!(sim, new_state, dt, forces, current_time)
    Jutul.reset_variables!(sim.storage, sim.model, new_state)
    Jutul.update_secondary_variables!(sim.storage, sim.model)
    Jutul.update_before_step!(sim.storage, sim.model, dt, forces, time=current_time)
    Jutul.update_linearized_system!(sim.storage, sim.model)
    return sim
end

"""
    forward_rule_jvp(hat_m, sim_forward, sim_backward, sim_param, states; 
                     opt_config_u=nothing, state_ref=nothing)

Computes the forward–mode Jacobian–vector product (JVP) for a multi–time step simulator.
For time step 1, it computes
  Q₁ = - A₁⁻¹ · C₁   and   x₁ = Q₁ * hat_m,
and then for n ≥ 2:
  Qₙ = - Aₙ⁻¹ · (Bₙ * Qₙ₋₁ + Cₙ)   and   xₙ = Qₙ * hat_m.
"""
function forward_rule_jvp(hat_m, sim_forward, sim_backward, sim_param, states, tstep, forces; 
                           opt_config_u=nothing, state_ref=nothing)
    N = length(states)
    x = Vector{Vector{Float64}}(undef, N)

    # ----- Time step 1 -----
    state1 = states[1]
    if isa(state1, AbstractVector)
        if isnothing(opt_config_u) || isnothing(state_ref)
            error("Vectorized state provided but opt_config_u or state_ref is missing.")
        end
        state1 = deepcopy(state_ref)
        targets = Jutul.optimization_targets(opt_config_u, sim_forward.model)
        mapper, = Jutul.variable_mapper(sim_forward.model, :primary; targets, config=opt_config_u)
        devectorize_variables!(state1, sim_forward.model, states[1], mapper, config=opt_config_u)
    end
    # For time step 1, update simulators
    update_simulator!(sim_forward, state1, tstep, forces, 1) #TODO: is current_time correct?
    update_simulator!(sim_backward, state1, tstep, forces, 1)
    update_simulator!(sim_param, state1, tstep, forces, 1)

    # Extract operators (assumed to be obtained via linear_operator)
    A1 = linear_operator(sim_forward.storage.LinearizedSystem) # Jutul.LinearizedSystem{Jutul.EquationMajorLayout, SparseMatrixCSC{Float64, Int64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    C1 = linear_operator(sim_param.storage.LinearizedSystem)

    # x[1] = - (A1 \ C1) * hat_m
    rhs = C1 * hat_m           # Vector result
    x0_guess = zeros(eltype(rhs), length(rhs))
    # Solve A1 * x_sol ≈ rhs with restarted GMRES:
    # - restart = 50 basis vectors
    gmres!(x0_guess, A1, rhs; restart=50, reltol=1e-9, maxiter=500)
    x[1] = - x0_guess
    println("x[1]", x0_guess, rhs)

    # ----- Subsequent time steps -----
    for n in 2:N
        state_n = states[n]
        if isa(state_n, AbstractVector)
            if isnothing(opt_config_u) || isnothing(state_ref)
                error("Vectorized state provided but opt_config_u or state_ref is missing.")
            end
            state_n = deepcopy(state_ref)
            targets = Jutul.optimization_targets(opt_config_u, sim_forward.model)
            mapper, = Jutul.variable_mapper(sim_forward.model, :primary; targets, config=opt_config_u)
            devectorize_variables!(state_n, sim_forward.model, states[n], mapper, config=opt_config_u)
        end
        update_simulator!(sim_forward, state_n, tstep, forces, n-1)
        update_simulator!(sim_backward, state_n, tstep, forces, n-1)
        update_simulator!(sim_param, state_n, tstep, forces, n-1)

        A_n = linear_operator(sim_forward.storage.LinearizedSystem)
        B_n = linear_operator(sim_backward.storage.LinearizedSystem)
        C_n = linear_operator(sim_param.storage.LinearizedSystem)
        
        rhs_n = (B_n * x[n-1] + C_n * hat_m)
        xn_guess = zeros(eltype(rhs_n), length(rhs_n))
        gmres!(xn_guess, A_n, rhs_n; restart=50, reltol=1e-9, maxiter=500)
        print("xn_guess", xn_guess)
        x[n] = - xn_guess
    end

    return x
end

########################################################################
# 2. Test Problem Setup and Objective Definition
########################################################################

function setup_bl(; nc = 100, time = 1.0, nstep = 100, poro = 0.1, perm = 9.8692e-14)
    T = time
    tstep = repeat([T/nstep], nstep)
    G = get_1d_reservoir(nc, poro = poro, perm = perm)
    nc = Jutul.number_of_cells(G)
    bar = 1e5
    p0 = 1000 * bar
    sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()))
    model = SimulationModel(G, sys)
    model.primary_variables[:Pressure] = JutulDarcy.Pressure(minimum=-Inf, max_rel=nothing)
    kr = BrooksCoreyRelativePermeabilities(sys, [2.0, 2.0])
    Jutul.replace_variables!(model, RelativePermeabilities=kr)
    tot_time = sum(tstep)
    parameters = Jutul.setup_parameters(model, PhaseViscosities=[1e-3, 5e-3])
    state0 = Jutul.setup_state(model, Pressure=p0, Saturations=[0.0, 1.0])
    irate = 100 * sum(parameters[:FluidVolume]) / tot_time
    src = [SourceTerm(1, irate, fractional_flow=[1.0-1e-3, 1e-3]),
           SourceTerm(nc, -irate, fractional_flow=[1.0, 0.0])]
    forces = Jutul.setup_forces(model, sources=src)
    return (model, state0, parameters, forces, tstep)
end

# Set up the reference and test cases.
N = 100; Nt = 100; poro_ref = 0.1; perm_ref = 9.8692e-14

model_ref, state0_ref, parameters_ref, forces, tstep = setup_bl(nc=N, nstep=Nt, poro=poro_ref, perm=perm_ref)
states_ref, = simulate(state0_ref, model_ref, tstep, parameters=parameters_ref, forces=forces, info_level=-1)

model, state0, parameters = setup_bl(nc=N, nstep=Nt, poro=2*poro_ref, perm=1.0*perm_ref)[1:3]
output = simulate(state0, model, tstep, parameters=parameters, forces=forces, info_level=-1)
states, rep = output
println("states", size(states))
println("rep", keys(rep))

# Define a per–time–step misfit function.
function mass_mismatch(m, state, dt, step_no, forces)
    state_ref = states_ref[step_no]
    fld = :Saturations
    val = state[fld]
    ref = state_ref[fld]
    err = 0.0
    for i in axes(val, 2)
        err += (val[1, i] - ref[1, i])^2
    end
    fld = :Pressure
    val = state[fld]
    ref = state_ref[fld]
    for i in axes(val, 2)
        err += (val[i] - ref[i])^2
    end
    return dt * err
end

# Define the overall objective (summing over time steps).
function objective(output)
    states, rep = output
    misfit = Jutul.evaluate_objective(mass_mismatch, model, states, tstep, forces)
    return misfit
end
@assert objective((states_ref, nothing)) == 0.0
@assert objective(output) > 0.0
println("Passed objective test.")

########################################################################
# 3. Objective Wrappers and Helper for the State Sensitivity
########################################################################

# Given a parameter vector x, run the adjoint-enabled simulation and compute the objective.
function full_objective_custom(x; opt_config_params)
    # output = simulate_ad(state0, model, tstep, x, forces;
                        #  parameters_ref=parameters, opt_config_params=opt_config_params, info_level=-1)
    output = simulate(state0, model, tstep, parameters=parameters, forces=forces, info_level=-1)
    return objective(output)
end

# Compute the gradient of mass_mismatch with respect to its state argument.
# The first argument is a dummy (since mass_mismatch does not use it) so we pass 0.
function compute_dmismatch_dstate(dummy, state, dt, step_no, forces)
    f(s) = mass_mismatch(dummy, s, dt, step_no, forces)
    return gradient(f, state)[1]
end

# Custom wrapper that computes the overall objective and its directional derivative via the custom JVP.
function full_objective_jvp(x, hat_x; opt_config_params)
    # 1. Run the simulation and compute the base objective.
    # output = simulate_ad(state0, model, tstep, x, forces;
                        #  parameters_ref=parameters, opt_config_params=opt_config_params, info_level=-1)
    output = simulate(state0, model, tstep, parameters=parameters, forces=forces, info_level=-1)
    base_obj = objective(output)
    base_states = output.states  # vector of simulation states over time

    # 2. Create simulators for the three modes.
    sim_forward  = Simulator(model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :forward, extra_timing = nothing)
    sim_backward = Simulator(model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :reverse, extra_timing = nothing)  # for ∂F/∂uₙ₋₁
    sim_param    = Simulator(model, state0 = deepcopy(state0), parameters = deepcopy(parameters), mode = :sensitivities, extra_timing = nothing)  # for ∂F/∂m

    # 3. Compute state sensitivities (the forward JVP) for each time step.
    state_jvp = forward_rule_jvp(hat_x, sim_forward, sim_backward, sim_param, base_states, tstep, forces;
                                 opt_config_u=opt_config_params, state_ref=base_states)

    # 4. Chain the state sensitivities with the derivative of mass_mismatch.
    d_obj = 0.0
    for (i, (state, st_jvp)) in enumerate(zip(base_states, state_jvp))
        dt = tstep[i]
        # We use 0 as a dummy value for m.
        d_mass = compute_dmismatch_dstate(0, state, dt, i, forces)
        d_obj += dot(d_mass, st_jvp)
    end

    return base_obj, d_obj
end

########################################################################
# 4. Finite-Difference Test for the JVP
########################################################################

"""
    jvp_test_obj(x0, hat_x; opt_config_params, h0, hfactor, maxiter)

For a given base parameter vector x0 and perturbation hat_x, this function:
  1. Computes the base objective and the directional derivative (via full_objective_jvp)
  2. Computes a finite-difference (FD) estimate of the directional derivative for various step sizes h.
  3. Prints a table comparing the FD derivative, custom JVP, and the absolute error.
"""
# function jvp_test_obj(x0, hat_x; opt_config_params, h0=5e-2, hfactor=0.8, maxiter=6)
#     base_val, jvp_val = full_objective_jvp(x0, hat_x; opt_config_params=opt_config_params)
#     println("       h         FD Deriv         Custom JVP      FD Error")
#     h = h0
#     for iter in 1:maxiter
#         fd_deriv = (full_objective_custom(x0 + h * hat_x; opt_config_params=opt_config_params) - base_val) / h
#         err = abs(fd_deriv - jvp_val)
#         @printf("%10.3e   %14.6e   %14.6e   %14.6e\n", h, fd_deriv, jvp_val, err)
#         h *= hfactor
#     end
# end

function jvp_test_obj(x0, hat_x; opt_config_params, h0=5e-2, hfactor=0.8, maxiter=6)
    base_val, jvp_val = full_objective_jvp(x0, hat_x; opt_config_params=opt_config_params)

    println("      h        FD Deriv       JVP       |FD - JVP|   |Residual| = |F(x+εv) - F(x) - εJVP|")
    h = h0
    for iter in 1:maxiter
        f_eps = full_objective_custom(x0 + h * hat_x; opt_config_params=opt_config_params)
        fd_deriv = (f_eps - base_val) / h
        residual = abs(f_eps - base_val - h * jvp_val)
        err = abs(fd_deriv - jvp_val)

        @printf("%10.3e   %14.6e   %10.6e   %12.4e   %14.6e\n", h, fd_deriv, jvp_val, err, residual)
        h *= hfactor
    end
end


# ## Set up parameter optimization
#
# This gives us a set of function handles together with initial guess and limits.
# Generally calling either of the functions will mutate the data Dict. The options are:
# F_o(x) -> evaluate objective
# dF_o(dFdx, x) -> evaluate gradient of objective, mutating dFdx (may trigger evaluation of F_o)


# F_and_dF(F, dFdx, x) -> evaluate F and/or dF. Value of nothing will mean that the corresponding entry is skipped.
print_obj = 100
# Also, assume opt_config_params (here named cfg) is defined in your optimization configuration.
cfg = optimization_config(model, parameters)
# (Adjust the configuration as in your code below if necessary.)
for (ki, vi) in cfg
    if ki in [:TwoPointGravityDifference, :PhaseViscosities]
        vi[:active] = false
    end
    if ki == :Transmissibilities
        vi[:scaler] = :default
    end
end
opt_info = setup_parameter_optimization(model, state0, parameters, tstep, forces, mass_mismatch, cfg, print = print_obj, param_obj = true);
F_o, dF_o, F_and_dF, x0, lims, data = opt_info
F_initial = F_o(x0)
dF_initial = dF_o(similar(x0), x0)
# dx = fill(0.01, length(x0))  # ✅ A single perturbation vector

dx = fill(0.01, 200)


@info "Initial objective: $F_initial, gradient norm $(norm(dF_initial))"

# Now run the finite-difference test of the JVP:
jvp_test_obj(x0, dx; opt_config_params=cfg, h0=5e-2, hfactor=0.8, maxiter=6)






