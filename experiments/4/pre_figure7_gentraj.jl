using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using UnPack

RESULTS_DIR = joinpath(@__DIR__, "results")

prob, (tsteps, ode_data), noise_var, θ_init, θ_bounds, u0_bounds =
    pendulum(tspan=(0.0, 13.0))
proj = [0 1]
ode_data = [proj * u for u in ode_data]
true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10);
D = length(prob.u0)
noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data];

plot(true_sol, color=:black, linestyle=:dash, label="")
scatter!(tsteps, ProbNumDiffEq.stack(noisy_ode_data), color=(1:D)', label="")

trajectories = []
for p0 in (0.5, 2.0, 5.0, 9.0)
    trajectory = []
    plot_callback = function ()
        j = 0
        function (x, l, times, states)
            j += 1

            p = x[1]
            σ² = exp(x[2])
            κ² = exp(x[3])

            @info "[Iteration $j] Callback" p σ² κ² l
            push!(trajectory, (p=p, diffusion=κ²))

            plt = scatter(
                tsteps,
                ProbNumDiffEq.stack(noisy_ode_data),
                label="",
                color=(1:D)',
                markersize=2,
                markerstrokewidth=0.1,
            )
            plot!(true_sol, color=:black, linestyle=:dot, label="")

            smoothed_means = ProbNumDiffEq.stack([x.μ for x in states])
            smoothed_stds = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states])
            plot!(
                times,
                smoothed_means,
                ribbon=2smoothed_stds,
                fillalpha=0.1,
                color=[2 1],
                label="",
                marker=:o,
                markersize=1,
                markerstrokewidth=0.1,
            )

            plot!(ylims=(-2, 2))

            display(plt)

            return false
        end
    end

    κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps, proj)
    u0 = noisy_ode_data[1]
    u0 = clamp.(u0, u0_bounds[1], u0_bounds[2])
    σ²0 = 1.0

    θ_start = [p0, log(σ²0), log(κ²0)]

    opt_params = (
        prob=prob,
        ode_data=noisy_ode_data,
        tsteps=tsteps,
        dt=1e-1,
        # dt=false,
        proj=proj,
    )

    function loss(x, other_params)
        p, σ²_log, κ²_log = x
        @unpack prob, ode_data, tsteps, dt = other_params
        data = (t=tsteps, u=noisy_ode_data)
        return exact_nll(
            remake(
                prob,
                # u0=u0,
                p=p,
            ),
            data,
            exp(σ²_log),
            # noise_var,
            exp(κ²_log);
            dt=dt,
            proj=proj,
        )
    end
    l, times, states = loss(θ_start, opt_params)

    ############################################################################################
    # First optimize only the diffusion!
    ############################################################################################
    optimizer = LBFGS()
    # optimizer = ADAM(1.0)
    # optimizer = BFGS()

    diff_loss(kappa, other_params) = loss([θ_start[1], θ_start[2], kappa[1]], other_params)
    f = OptimizationFunction(diff_loss, GalacticOptim.AutoForwardDiff())
    optprob =
        OptimizationProblem(f, [log(κ²0)], opt_params; lb=[log(1e-20)], ub=[log(1e50)])
    _cb = (x, l, args...) -> begin
        @info "Diffusion Opt CB" κ² = exp(x[1]) loss = l
        push!(trajectory, (p=p0, diffusion=exp(x[1])))
        return false
    end
    optsol = solve(optprob, optimizer; maxiters=100, cb=_cb)

    x0 = [θ_start[1], θ_start[2], optsol.u[1]]
    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(
        f,
        x0,
        opt_params;
        lb=[0.0, log(1e-8), log(1e-20)],
        ub=[100.0, log(1e4), log(1e50)],
    )
    optimizer = LBFGS()
    optsol = solve(optprob, optimizer; maxiters=1000, cb=plot_callback())
    @info "DONE"

    push!(trajectories, (p0=p0, trajectory=copy(trajectory)))
end

using JLD
save(joinpath(RESULTS_DIR, "trajectories.jld"), "trajectories", trajectories)
