using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, LinearAlgebra, Statistics
using Fenrir


RESULTS_DIR = joinpath(@__DIR__, "results")

PROBLEMS = (
    (lotkavolterra, "lv"),
    (fitzhughnagumo, "fhn"),
    # (protein_transduction, "pt"),
)
ALG = Dict(
    "lv" => Tsit5,
    "fhn" => RadauIIA5,
    "pt" => Tsit5,
)

for (problem, probname) in PROBLEMS, noisestr in ("low", "high")

    Alg = ALG[probname]

    prob, (tsteps, ode_data), noise_levels, θ_init, θ_bounds, u0_bounds = problem()
    D = length(ode_data[1])
    noise_var = noise_levels[noisestr]

    filename = "$(probname)_$(noisestr)noise_classic.txt"
    filepath = joinpath(RESULTS_DIR, filename)
    run(`rm -f $filepath`)

    filename2 = "$(probname)_$(noisestr)noise_classic_withiv.txt"
    filepath2 = joinpath(RESULTS_DIR, filename2)
    run(`rm -f $filepath2`)

    @info "save into" filepath

    for i in 1:100
        @info "[$i]"

        # Generate the noisy data
        noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data]

        function loss(p, opt_params)
            _u0 = p[1:D]
            _p = p[D+1:end]
            sol = solve(
                remake(prob, u0=_u0, p=_p),
                Alg(),
                # abstol=1e-6, reltol=1e-3,
                abstol=1e-8, reltol=1e-6,
                saveat=tsteps,
            )
            if sol.retcode != :Success
                @warn "Solver failed!"
                return 1e6
            end
            return sum(norm.(sol.u - noisy_ode_data) .^ 2)
            # return mean(norm.(sol.u - noisy_ode_data))
        end

        # Initial parameters, with both vector-field parameters and the log-diffusion
        u0 = min.(max.(noisy_ode_data[1], u0_bounds[1]), u0_bounds[2])
        p0 = θ_init()
        θ_start = [u0..., p0...]
        # @info "[$i] Initial parameter" u0 p0

        @info "[$i] Perform testrun"
        l = loss(θ_start, nothing)

        f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
        optprob = OptimizationProblem(
            f, θ_start;
            lb=[u0_bounds[1]..., θ_bounds[1]...],
            ub=[u0_bounds[2]..., θ_bounds[2]...],
        )

        @info "[$i] Optimize"
        optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking());
        cb() = (j=0; (p, l, args...)-> (j+=1; @info "[$i] Iter $j" l p; false))
        optsol = solve(optprob, optimizer; maxiters=1000,
                       # cb=cb()
                       )
        opt_u0, opt_p = optsol.u[1:D], optsol.u[D+1:end]
        @info "[$i] Done" opt_u0 opt_p

        open(filepath, "a") do f
            write(f, join(string.(opt_p), ",") * "\n")
        end
        open(filepath2, "a") do f
            write(f, join(string.([opt_u0..., opt_p...,]), ",") * "\n")
        end
    end
end
