using Oceananigans
using Logging
using Statistics


include("rbc_sim3D.jl")

# Global variables to hold simulation state
global simulation = nothing
global model = nothing
global step = 1
global time = 0.0

"""
Initialize a Rayleigh-Bénard simulation with the given parameters
"""
function initialize_simulation(; Ra=2500, Pr=0.7, L=[4*pi, 4*pi, 2], grid=[32, 32, 16], T_diff=[1, 2], heaters=8, heater_limit=0.9, dt=0.125, dt_solver=0.01, seed=42, checkpoint_path=nothing, use_gpu=false)
    oceananigans_logger = Oceananigans.Logger.OceananigansLogger(
        stdout,
        Logging.Warn;
        show_info_source=true
    )
    global_logger(oceananigans_logger)

    # Setup simulation parameters
    global N = grid
    global domain = L
    global min_b = T_diff[1]
    global Δb = T_diff[2] - T_diff[1]
    global actuators = (heaters, heaters)
    global actuator_limit = heater_limit
    global Δt = dt 

    random_kick = 0.01
    Δt_solver = dt_solver

    ν = sqrt(Pr / Ra)
    global κ = 1 / sqrt(Pr * Ra)

    # simulation is done in free-fall time units
    # t_ff = H/U_ff = H/sqrt(gαΔTH) = H/(1/H) = H^2
    # since computation of ν,κ above assumes that gαΔTH^3=1 ⇔ sqrt(gαΔTH) = 1/H
    global t_ff = L[3]^2

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Initialize action
    global action = zeros(actuators...)

    # Initialize simulation components
    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, v_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    # Create model
    global model = define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
    # Initialize model
    if isnothing(checkpoint_path)
        initialize_model(model, min_b, L[3], Δb, random_kick)
    else
        initialize_from_checkpoint(model, checkpoint_path)
    end

    # Setup simulation
    global simulation = Simulation(model, Δt=Δt_solver * t_ff, stop_time=Δt * t_ff)
    simulation.verbose = false

    # Reset state
    global step = 1
    global time = 0.0

end

"""
Step the simulation forward by one timestep
"""
function step_simulation(actuation)
    global simulation, model, Δt, t_ff, N, step, time

    if simulation === nothing
        error("Simulation not initialized. Call initialize_simulation first.")
    end

    # preprocess action
    global action = preprocess_action(actuation)

    # Run for one step
    run!(simulation)
    simulation.stop_time += Δt * t_ff

    time += Δt * t_ff
    step += 1

    # Check for NaNs
    contains_nans = step_contains_NaNs(model, N)
    if contains_nans
        return false
    end

    return true
end

"""
Get the current state of the simulation
"""
function get_state()
    global model, N

    if model === nothing
        error("Simulation not initialized. Call initialize_simulation first.")
    end

    # Extract fields from the current model
    state = zeros(6, N[1], N[2], N[3])
    state[1, :, :, :] = model.tracers.b[1:N[1], 1:N[2], 1:N[3]]
    state[2, :, :, :] = model.velocities.u[1:N[1], 1:N[2], 1:N[3]]
    state[3, :, :, :] = model.velocities.v[1:N[1], 1:N[2], 1:N[3]]
    state[4, :, :, :] = model.velocities.w[1:N[1], 1:N[2], 1:N[3]]
    state[5, :, :, :] = model.pressures.pHY′[1:N[1], 1:N[2], 1:N[3]]
    state[6, :, :, :] = model.pressures.pNHS[1:N[1], 1:N[2], 1:N[3]]

    return state
end

"""
Get the current simulation time
"""
function get_info()
    global time, step
    return (time, step)
end

"""
Get nusselt number
"""
function get_nusselt()
    # Definition by Vasanth et al 2024
    global κ
    global Δb
    
    data = get_state()

    # Extract buoyancy (tracer `b`) and vertical velocity `w`
    b = data[1, :, :, :]
    w = data[4, :, :, :]

    # get conductive reference temperature field
    # build conduction profile
    Nz = size(b, 3) # number of grid points in vertical direction
    dz = 1 / Nz
    z = dz/2 : dz : 1 - dz/2  # midpoints of grid cells in vertical direction
    T_conductive = (1 .- z) .* Δb .+ min_b
    T_conductive = reshape(T_conductive, (1, 1, Nz))
    
    Tprime = b .- T_conductive  # temperature deviation from conductive profile

    # Convective flux: ⟨w b⟩
    q_conv = mean(Tprime .* w)        # full‑volume average
    
    return 1 + (q_conv / κ )
end

"""
Shutdown the RBC simulation and free memory
"""
function shutdown_simulation()
    global simulation = nothing
    global model = nothing

    GC.gc()  # Force Julia garbage collection
end


# Export functions
export initialize_simulation, step_simulation, get_state, shutdown_simulation
