using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using UnPack

# Set here what's supposed to run
# probname = "lv"
# probname = "fhn"
# probname = "pt"
# noisestr = "low"
# noisestr = "high"
@info "STARTING THE EXPERIMENT:" probname noisestr


RESULTS_DIR = joinpath(@__DIR__, "results")


PROBLEMS = Dict(
    "lv" => lotkavolterra,
    "fhn" => fitzhughnagumo,
    "pt" => protein_transduction,
)

ALG = EK1
DT_CHOICES = Dict(
    "lv" => 5e-3,
    "fhn" => 5e-2,
    "pt" => false,
)
ORDER_CHOICES = Dict(
    "lv" => 5,
    "fhn" => 3,
    "pt" => 5,
)

problem = PROBLEMS[probname]

prob, (tsteps, ode_data), noise_levels, θ_init, θ_bounds, u0_bounds = problem()
D = length(ode_data[1])
P = length(prob.p)
noise_var = noise_levels[noisestr]
E = length(noise_var)

filename = "$(probname)_$(noisestr)noise_fenrir.txt"
filepath = joinpath(RESULTS_DIR, filename)
filename2 = "$(probname)_$(noisestr)noise_fenrir_withiv.txt"
filepath2 = joinpath(RESULTS_DIR, filename2)
# @info "Remove previous results" filepath filepath2
# run(`rm -f $filepath`)
# run(`rm -f $filepath2`)

@info "New problem & config" probname noisestr filepath

for i in 1:100

    # Generate the noisy data
    noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data]

    # Build the named tuple that will be used by the NLL loss function
    opt_params = (
        prob=prob,
        ode_data=noisy_ode_data,
        tsteps=tsteps,
        noise=noise_var,
        dt=DT_CHOICES[probname],
        order=ORDER_CHOICES[probname],
        tstops = probname == "pt" ?
            # union(0:5e-2:1, 1:0.1:2, 2:0.25:4, 4:0.5:10, 10:1:20, 20:2:100) : [],
            (10.0 .^ (-2.5:0.05:2)) : [],
    )

    # Initial parameters, with both vector-field parameters and the log-diffusion
    κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps) # * ones(E)
    u0 = clamp.(noisy_ode_data[1], u0_bounds[1], u0_bounds[2])
    p0 = θ_init()
    σ²0 = 1.0 # * ones(E)
    x0 = [u0..., p0..., log.(σ²0)..., log.(κ²0)...]

    σ²_bounds = (log(1e-4)*ones(length(σ²0)), log(1e2)*ones(length(σ²0)))
    κ²_bounds = (log(1e-20)*ones(length(κ²0)), log(1e50)*ones(length(κ²0)))

    vec2args(x) = (
        u0 = x[1:D],
        p = x[D+1:D+P],
        # σ² = exp.(x[D+P+1:D+P+E]),
        # κ² = exp.(x[D+P+E+1:end]),
        σ² = exp.(x[end-1]),
        κ² = exp.(x[end]),
    )


    @info "[$i] Testrun" x0
    function loss(x, other_params)
        @unpack u0, p, σ², κ² = vec2args(x)
        @unpack order, prob, ode_data, tsteps, dt, tstops = other_params
        data = (t=tsteps, u=noisy_ode_data)
        return exact_nll(remake(prob, u0=u0, p=p), data, σ², κ²; dt=dt, tstops=tstops, order=order)
    end
    l, times, states = loss(x0, opt_params);


    function loss_k(gp_params, other_params)
        u0, p = u0, p0
        # σ² = exp.(gp_params[1:E])
        # κ² = exp.(gp_params[E+1:end])
        σ² = exp.(gp_params[end-1])
        κ² = exp.(gp_params[end])
        p = one(κ²) .* p
        # p = one(eltype(κ²)) .* p
        @unpack prob, ode_data, tsteps, dt, tstops = other_params
        data = (t=tsteps, u=noisy_ode_data)
        return exact_nll(remake(prob, u0=u0, p=p), data, σ², κ²; dt=dt, tstops=tstops)
    end
    function _cb1()
        j = 0
        function CB1(x, l, args...)
            j+=1
            @info "[$i] Diff-Optim Iteration $j" loss=l σ²=exp.(x[1:E]) κ²=exp.(x[E+1:end])
            return j > 1000
        end
    end
    @info "[$i] First optimize for the diffusion"
    gp_x0 = [log.(σ²0)..., log.(κ²0)...]
    loss_k(gp_x0, opt_params)
    f = OptimizationFunction(loss_k, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(
        f, gp_x0, opt_params;
        lb=[σ²_bounds[1]..., κ²_bounds[1]...],
        ub=[σ²_bounds[2]..., κ²_bounds[2]...],
    )
    # optsol = solve(optprob, BBO_adaptive_de_rand_1_bin(); maxiters=1000, cb=_cb1())
    optsol = solve(optprob,
                   LBFGS();
                   # LBFGS(linesearch=Optim.LineSearches.BackTracking());
                   maxiters=200, cb=_cb1())
    x0 = [x0[1:D+P]..., optsol.u...]

    @info "[$i] Now optimize the full problem"
    function _cb2()
        j = 0
        function CB2(x, l, args...)
            j+=1
            @unpack u0, p, σ², κ² = vec2args(x)
            @info "[$i] Optim Iteration $j" loss=l u0 u0_true=prob.u0 p p_true=prob.p σ² κ²
            return j > 1000
        end
    end
    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(
        f, x0, opt_params;
        lb=[u0_bounds[1]..., θ_bounds[1]..., σ²_bounds[1]..., κ²_bounds[1]...],
        ub=[u0_bounds[2]..., θ_bounds[2]..., σ²_bounds[2]..., κ²_bounds[2]...],
    )
    optsol = solve(optprob,
                   LBFGS(linesearch=Optim.LineSearches.BackTracking());
                   # LBFGS();
                   maxiters=400, cb=_cb2(),
                   reltol=1e-6,
                   )
    @unpack u0, p, σ², κ² = vec2args(optsol.u)
    opt_u0, opt_p, opt_σ², opt_κ² = u0, p, σ², κ²
    @info "[$i] Done" prob.p opt_p opt_u0 opt_σ² opt_κ²;

    open(filepath, "a") do f
        write(f, join(string.(opt_p), ",") * "\n")
    end
    open(filepath2, "a") do f
        write(f, join(string.([opt_u0..., opt_p...,]), ",") * "\n")
    end
end
