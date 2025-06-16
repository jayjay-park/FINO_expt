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
using Flux: withgradient


########################################################################
# 2. Test Problem Setup and Objective Definition
########################################################################
function test_config()
    n = (30, 1, 15)
    d = (30.0, 30.0, 30.0)

    ## permeability
    K0 = 40 * md * ones(n)
    ϕ = 0.25
    K = deepcopy(K0)
    K[:,:,1:2:end] .*= 40

    model0 = jutulModel(n, d, ϕ, K1to3(K0))
    model = jutulModel(n, d, ϕ, K1to3(K))

    ## simulation time steppings
    tstep = 50 * ones(10)

    ## injection & production
    inj_loc = (15, 1, 10) .* d
    prod_loc = (30, 1, 10) .* d
    irate = 5e-3
    q = jutulForce(irate, inj_loc)
    q1 = jutulSource(irate, [inj_loc])
    q2 = jutulVWell(irate, inj_loc[1:2]; startz = 9 * d[3], endz = 11 * d[3])
    state0 = jutulState(JutulDarcyRules.setup_well_model(model, q, tstep)[3])
    state1 = JutulDarcyRules.setup_simple_model(model, q1, tstep)[3]
    return model, model0, q, q1, q2, state0, state1, tstep
end

model, model0, q, q1, q2, state0, state1, tstep = test_config();

# two diff permeability model: model, model0
# two reservoir simulation: state0, state1 (simpler model)


## set up modeling operator
S0 = jutulModeling(model0, tstep)
S = jutulModeling(model, tstep)

## simulation
x = log.(KtoTrans(CartesianMesh(model), model.K))
x0 = log.(KtoTrans(CartesianMesh(model0), model0.K))

ϕ = S.model.ϕ
ϕ0 = S0.model.ϕ

function misfit(x0, ϕ0, q, states_ref)
    states = S0(x0, ϕ0, q; config)
    sat_misfit = 0.5 * sum(sum((s[:Reservoir][:Saturations][1, :] .- sr[:Reservoir][:Saturations][1, :]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    pres_misfit = 0.5 * sum(sum((s[:Reservoir][:Pressure] .- sr[:Reservoir][:Pressure]) .^ 2) for (s, sr) in zip(states.states, states_ref.states))
    # sat_misfit
    sat_misfit + pres_misfit * 1e-14
end

rng = MersenneTwister(2023)

function sample_dx()
    dx = randn(rng, length(x0))
    dx = dx/norm(dx) * norm(x0)/5.0
end

# perturbation in permeability
dx = sample_dx()
# perturbation in porosity
ϕmask = ϕ .< 1
function sample_dϕ()
    ϕfactor = randn(rng, model.n[1:2:3])
    kernel = [0.25, 0.5, 0.25]
    kernel = kernel * kernel'
    kernel_idx = CartesianIndices(kernel) .- CartesianIndex((2, 2))
    dϕ = zeros(size(ϕfactor))
    for c in CartesianIndices(ϕfactor)
        if c.I[1] == 1 || c.I[1] == model.n[1] || c.I[2] == 1 || c.I[2] == model.n[3]
            continue
        end
        c_kernel = kernel_idx .+ c
        dϕ[c_kernel] .= kernel * ϕfactor[c_kernel]
    end
    dϕ = vec(dϕ)
    dϕ[.!ϕmask] .= 0
    # dϕ[ϕmask] = ϕ[ϕmask] .* exp.(ϕfactor[ϕmask])
    dϕ[ϕmask] = ϕfactor[ϕmask]/norm(ϕfactor[ϕmask]) * norm(ϕ[ϕmask])
end
dϕ = sample_dϕ()

# groundtruth state
@time states_ref, case, sim, x0_0 = S(x, ϕ, q; return_extra=true)

config = JutulDarcy.simulator_config(sim)
print("config", config)
for m in Jutul.submodels_symbols(case.model)
    config[:tolerances][m][:default] = 1e-10
end
config[:linear_solver].config.relative_tolerance = 1e-10
config[:info_level] = -1

@time states_ref = S(x, ϕ, q; config)
v_initial = misfit(x0, ϕ0, q, states_ref)
@show v_initial

misfit_dx = x0->misfit(x0, ϕ, q, states_ref)
misfit_dϕ = ϕ0->misfit(x, ϕ0, q, states_ref)
misfit_dboth = (x0,ϕ0)->misfit(x0, ϕ0, q, states_ref)

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
    N = length(tstep)
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
    update_simulator!(sim_forward, state1, tstep, forces, 1)
    update_simulator!(sim_backward, state1, tstep, forces, 1)
    update_simulator!(sim_param, state1, tstep, forces, 1)

    # Extract operators (assumed to be obtained via linear_operator)
    A1 = linear_operator(sim_forward.storage.LinearizedSystem) # Jutul.LinearizedSystem{Jutul.EquationMajorLayout, SparseMatrixCSC{Float64, Int64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    C1 = linear_operator(sim_param.storage.LinearizedSystem)

    # x[1] = - (A1 \ C1) * hat_m
    rhs = C1 * hat_m           # Vector result
    print("rhs", rhs)
    x0_guess = zeros(eltype(rhs), length(rhs))
    # Solve A1 * x_sol ≈ rhs with restarted GMRES:
    # - restart = 50 basis vectors
    gmres!(x0_guess, A1, rhs; restart=50, reltol=1e-9, maxiter=500)
    x[1] = - x0_guess
    println("x[1]", x0_guess)

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
        # print("rhs", rhs_n)
        xn_guess = zeros(eltype(rhs_n), length(rhs_n))
        gmres!(xn_guess, A_n, rhs_n; restart=50, reltol=1e-9, maxiter=500)
        # print("xn_guess", xn_guess)
        x[n] = - xn_guess
    end

    return x
end



########################################################################
# 3. Objective Wrappers and Helper for the State Sensitivity
########################################################################


function full_objective_jvp(x0, dx; config, sim, case0)
    # 1. Run the simulation and compute the base objective.
    base_obj = misfit_dx(x0)

    # 2. Create simulators for the three modes. Type: Jutul.Simulator
    sim_forward  = Simulator(case0, mode = :forward, extra_timing = nothing)
    sim_backward = Simulator(case0, mode = :reverse, extra_timing = nothing)  # for ∂F/∂uₙ₋₁
    sim_param    = Simulator(case0, mode = :sensitivities, extra_timing = nothing)  # for ∂F/∂m

    # 3. Compute state sensitivities (the forward JVP) for each time step.
    state_jvp = forward_rule_jvp(dx, sim_forward, sim_backward, sim_param, states_ref, tstep, forces;
                                 opt_config_u=config, state_ref=states_ref)

    # 4. Chain the state sensitivities with the derivative of mass_mismatch.
    d_obj = 0.0
    for (i, (state, st_jvp)) in enumerate(zip(states_ref, state_jvp))
        dt = tstep[i]
        # We use 0 as a dummy value for m.
        d_mass = misfit(x0 + dx)
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

function jvp_test_obj(x0, hat_x; config, sim, case, h0=5e-2, hfactor=0.8, maxiter=6)
    base_val, jvp_val = full_objective_jvp(x0, hat_x; config, sim, case0)

    println("      h        FD Deriv       JVP       |FD - JVP|   |Residual| = |F(x+εv) - F(x) - εJVP|")
    h = h0
    for iter in 1:maxiter
        f_eps = misfit_dx(x0 + h * hat_x) # computing misfit in output btw S and S0
        fd_deriv = (f_eps - base_val) / h
        residual = abs(f_eps - base_val - h * jvp_val)
        err = abs(fd_deriv - jvp_val)

        @printf("%10.3e   %14.6e   %10.6e   %12.4e   %14.6e\n", h, fd_deriv, jvp_val, err, residual)
        h *= hfactor
    end
end

@info "Initial objective: $F_initial, gradient norm $(norm(dF_initial))"
@time states0, case0, sim0, x0_0 = S0(x0, ϕ, q; return_extra=true)
config0 = JutulDarcy.simulator_config(sim0)

# Now run the finite-difference test of the JVP:
jvp_test_obj(x0, dx; config=config0, sim=sim0, case=case0, h0=5e-2, hfactor=0.8, maxiter=6)