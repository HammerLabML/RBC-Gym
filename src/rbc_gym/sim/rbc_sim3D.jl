using Printf
import Random
using Oceananigans
using Statistics
using HDF5
using ArgParse
using ProgressMeter

# script directory
dirpath = string(@__DIR__)


function simulate_3d_rb(dir, seed, random_inits, Ra, Pr, N, L, b, random_kick, Δt, Δt_snap, duration, use_gpu)

    println("Simulating 3D Rayleigh-Bénard convection with parameters:")
    println("  Directory: $dir")
    println("  Seed: $seed")
    println("  Random Initializations: $random_inits")
    println("  Rayleigh Number (Ra): $Ra")
    println("  Prandtl Number (Pr): $Pr")
    println("  Grid Size (N): $N")
    println("  Domain Size (L): $L")
    println("  Temperature Difference (b): $b")
    println("  Random Kick: $random_kick")
    println("  Time Step (Δt): $Δt")
    println("  Snapshot Interval (Δt_snap): $Δt_snap")
    println("  Duration: $duration")

    ν = sqrt(Pr / Ra) # c.f. line 33: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard.py
    κ = 1 / sqrt(Pr * Ra) # c.f. line 37: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard.py

    # simulation is done in free-flow time units
    # t_ff = H/U_ff = H/sqrt(gαΔTH) = H/(1/H) = H^2
    # since computation of ν,κ above assumes that gαΔTH^3=1 ⇔ sqrt(gαΔTH) = 1/H
    t_ff = L[3]^2

    totalsteps = Int(div(duration, Δt_snap * t_ff))

    global min_b, Δb
    min_b = b[1]
    Δb = b[2] - b[1]

    global domain, action, actuators, actuator_limit
    domain = L
    actuators = (8, 8)
    actuator_limit = 0.9
    action = preprocess_action(zeros(actuators...))

    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, v_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    model = define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)

    if !isdir(dir)
        mkpath(dir)
    end
    path = joinpath(dir, "ckpt_ra$(Ra).h5")
    h5_file = h5open(path, "w")

    attrs(h5_file)["num_episodes"] = random_inits
    attrs(h5_file)["start_seed"] = seed
    db = create_dataset(h5_file, "b", datatype(Float64), dataspace(random_inits, size(model.tracers.b)...))
    du = create_dataset(h5_file, "u", datatype(Float64), dataspace(random_inits, size(model.velocities.u)...))
    dv = create_dataset(h5_file, "v", datatype(Float64), dataspace(random_inits, size(model.velocities.v)...))
    dw = create_dataset(h5_file, "w", datatype(Float64), dataspace(random_inits, size(model.velocities.w)...))


    for i ∈ 1:random_inits
        println("Simulating random initialization $(i)/$(random_inits)...")

        # Make sure that every random initialization is indeed independend of each other
        # (even when script is restarted)
        Random.seed!(seed + i)

        model = define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
        initialize_model(model, min_b, L[3], Δb, random_kick)

        simulation = Simulation(model, Δt=Δt * t_ff, stop_time=Δt_snap * t_ff)
        simulation.verbose = false

        success = simulate_model(simulation, model, Δt, t_ff, Δt_snap, totalsteps, N)

        if (!success)
            return
        end

        db[i, :, :, :] = model.tracers.b
        du[i, :, :, :] = model.velocities.u
        dv[i, :, :, :] = model.velocities.v
        dw[i, :, :, :] = model.velocities.w
    end

    # Save the simulation results to a file
    close(h5_file)
    println("Saved data to: ", h5_file)
end


function define_sample_grid(N, L, use_gpu)
    if use_gpu
        grid = RectilinearGrid(GPU(), size=N, x=(0, L[1]), y=(0, L[2]), z=(0, L[3]),
            topology=(Periodic, Periodic, Bounded))
    else
        grid = RectilinearGrid(size=(N), x=(0, L[1]), y=(0, L[2]), z=(0, L[3]),
            topology=(Periodic, Periodic, Bounded))
    end
    return grid
end


function preprocess_action(action)
    # According to Vasanth et al. (2024)
    global actuators, actuator_limit, min_b, Δb

    if size(action) != actuators
        error("Action size does not match the number of actuators. Expected $(actuators), got $(size(action)).")
    end

    # Subtract mean to center the action around zero
    action = action .- mean(action)

    # Get maximum absolute value of the action or 1
    K = max(1, maximum(abs.(action)))

    # Add perturbations scaled by maximum to the action
    a0 = fill(min_b + Δb, size(action))
    return a0 + (action ./ K) .* actuator_limit
end


function bottom_T(x, y, t)
    global action, domain, actuators

    nx, ny = actuators
    Lx, Ly = domain[1], domain[2]

    i = clamp(floor(Int, x / Lx * nx) + 1, 1, nx)
    j = clamp(floor(Int, y / Ly * ny) + 1, 1, ny)

    return action[i, j]
end


function define_boundary_conditions(min_b, Δb)
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(min_b),
        bottom=ValueBoundaryCondition(bottom_T))
    return u_bcs, v_bcs, b_bcs
end


function define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
    model = NonhydrostaticModel(; grid,
        advection=UpwindBiasedFifthOrder(),
        timestepper=:RungeKutta3,
        tracers=(:b),
        buoyancy=Buoyancy(model=BuoyancyTracer()),
        closure=(ScalarDiffusivity(ν=ν, κ=κ)),
        boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs,),
        coriolis=nothing
    )
    return model
end


function initialize_model(model, min_b, Lz, Δb, kick)
    # Set initial conditions
    uᵢ(x, y, z) = kick * randn()
    vᵢ(x, y, z) = kick * randn()
    wᵢ(x, y, z) = kick * randn()
    bᵢ(x, y, z) = clamp(min_b + (Lz - z) * Δb / 2 + kick * randn(), min_b, min_b + Δb)

    # Send the initial conditions to the model to initialize the variables
    set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=bᵢ)
end


function initialize_from_checkpoint(model, path)
    h5_file = h5open(path, "r")

    n = attrs(h5_file)["num_episodes"]
    idx = rand(1:n)

    bb = read(h5_file, "b")[idx,:,:,:]
    uu = read(h5_file, "u")[idx,:,:,:]
    vv = read(h5_file, "v")[idx,:,:,:]
    ww = read(h5_file, "w")[idx,:,:,:]

    println("Loading checkpoint with index: $idx from file: $path")
    set!(model, u = uu, v = vv, w = ww, b = bb)
    close(h5_file)
end


function simulate_model(simulation, model, Δt, t_ff, Δt_snap, totalsteps, N)
    cur_time = 0.0
    @showprogress 2 "Simulating..." for i in 1:totalsteps
        run!(simulation)
        simulation.stop_time += Δt_snap * t_ff
        cur_time += Δt_snap * t_ff

        if (step_contains_NaNs(model, N))
            printstyled("[ERROR] NaN values found!\n"; color=:red)
            return false
        end
    end

    return true
end


function step_contains_NaNs(model, N)
    contains_nans = (any(isnan, model.tracers.b[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.u[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.v[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.w[1:N[1], 1:N[2], 1:N[3]]))
    return contains_nans
end


function parse_arguments()
    s = ArgParseSettings(description="Simulates 3D Rayleigh-Bénard.")

    @add_arg_table s begin
        "--dir"
        help = "The path to the directory to store the simulations in."
        default = joinpath(dirpath, "data")
        "--seed"
        help = "Random seed for the simulation."
        arg_type = Int
        default = 42
        "--random_inits"
        help = "The number of random initializations to simulate"
        arg_type = Int
        default = 1
        "--Ra"
        help = "Rayleigh number"
        arg_type = Int
        default = 2500
        "--Pr"
        help = "Prandtl number"
        arg_type = Float64
        default = 0.7
        "--N"
        help = "The size of the grid [width, depth, height]"
        arg_type = Int
        nargs = 3
        default = [32, 32, 16]
        "--L"
        help = "The spatial dimensions of the domain [width, height]"
        arg_type = Float64
        nargs = 3
        default = [4 * pi, 4 * pi, 2]
        "--b"
        help = "The temperature of the bottom and top plate"
        arg_type = Float64
        nargs = 2
        default = [1, 2]
        "--random_kick"
        help = "The amplitude of the random initial perturbation (kick)"
        arg_type = Float64
        default = 0.01
        "--delta_t"
        help = "Time delta between simulated steps (in free fall time units)"
        arg_type = Float64
        default = 0.01
        dest_name = "Δt"
        "--delta_t_snap"
        help = "Time delta between saved snapshots (in free fall time units)"
        arg_type = Float64
        default = 0.125
        dest_name = "Δt_snap"
        "--duration"
        help = "The duration of a simulation"
        arg_type = Int
        default = 200
        "--use_cpu"
        help = "Runs the simulation on CPU when argument given."
        action = :store_false
        dest_name = "use_gpu"
    end

    return parse_args(s)
end

is_script = abspath(PROGRAM_FILE) == @__FILE__
if (is_script)
    args = parse_arguments()
    simulate_3d_rb(
        args["dir"],
        args["seed"],
        args["random_inits"],
        args["Ra"],
        args["Pr"],
        args["N"],
        args["L"],
        args["b"],
        args["random_kick"],
        args["Δt"],
        args["Δt_snap"],
        args["duration"],
        args["use_gpu"])
end