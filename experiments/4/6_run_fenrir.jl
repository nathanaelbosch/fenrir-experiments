using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using UnPack
using Random

prob, (tsteps, ode_data), noise_var, θ_init, θ_bounds, u0_bounds = pendulum(
    tsteps=0:0.01:10.0,
)
proj = [0 1]
ode_data = [proj*u for u in ode_data]
true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10);
D = length(prob.u0)
RNG = MersenneTwister(1234)
noisy_ode_data = [u + sqrt.(noise_var) .* randn(RNG, size(u)) for u in ode_data];

Plots.plot(true_sol, color=:black, linestyle=:dash, label="")
Plots.scatter!(tsteps, ProbNumDiffEq.stack(noisy_ode_data), color=(1:D)', label="")

# trajectory = []
plot_callback = function ()
    j = 0
    function (x, l, times, states)
        j += 1

        u0 = x[1:D]
        p = x[D+1]
        σ² = exp(x[D+2])
        κ² = exp(x[D+3])

        @info "[Iteration $j] Callback" p σ² κ² l
        # push!(trajectory, (p=p, diffusion=κ²))

        plt = Plots.scatter(tsteps, ProbNumDiffEq.stack(noisy_ode_data), label="",
                      color=(1:D)', markersize=2, markerstrokewidth=0.1)
        Plots.plot!(true_sol, color=:black, linestyle=:dot, label="")

        smoothed_means = ProbNumDiffEq.stack([x.μ for x in states])
        smoothed_stds = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states])
        Plots.plot!(times, smoothed_means, ribbon=2smoothed_stds,
              fillalpha=0.1,
              color=[2 1],
              label="",
              marker=:o, markersize=1, markerstrokewidth=0.1,
              )

        Plots.plot!(ylims=(-2, 2))

        display(plt)

        return false
    end
end

κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps, proj)
κ²0 = 1e30
u0 = [0.0, noisy_ode_data[1][1]]
# u0 = [0.0, 0.0]
u0 = clamp.(u0, u0_bounds[1], u0_bounds[2])
σ²0 = 1.0
p0 = θ_init()
p0 = 8.0


x0 = [u0..., p0, log(σ²0), log(κ²0)]

opt_params = (
    prob=prob,
    ode_data=noisy_ode_data,
    tsteps=tsteps,
    # dt=1e-1,
    dt=1e-3,
    # dt=false,
    proj=proj,
);

function loss(x, other_params)
    u0, p, σ²_log, κ²_log = x[1:D], x[D+1], x[D+2], x[D+3]
    @unpack prob, ode_data, tsteps, dt = other_params
    data = (t=tsteps, u=noisy_ode_data)
    return exact_nll(remake(prob,
                            u0=u0,
                            p=p), data,
                     exp(σ²_log),
                     # noise_var,
                     exp(κ²_log); dt=dt, proj=proj)
end
l, times, states = loss(x0, opt_params);
plot_callback()(x0, l, times, states)
# plot_fenrir!(ax11, (t=times, u=states), (t=tsteps, u=noisy_ode_data))


############################################################################################
# First optimize only the diffusion!
############################################################################################
# diff_loss(x, other_params) = loss([x0[1:end-2]..., x...], other_params)
# f = OptimizationFunction(diff_loss, GalacticOptim.AutoForwardDiff())
# optprob = OptimizationProblem(f, [log(σ²0), log(κ²0)], opt_params;
#                               lb=[log(1e-6), log(1e-20)], ub=[log(1e4), log(1e50)])
# _cb = (x, l, args...) -> begin
#     @info "Diffusion Opt CB" σ²=exp(x[1]) κ²=exp(x[2]) loss=l
#     # push!(trajectory, (p=p0, diffusion=exp(x[1])))
#     return false
# end
# optsol = solve(optprob, LBFGS(); maxiters=100, cb=_cb)
# @info "Use these parameters to start the full optimization"
# x0 = [x0[1:end-2]..., optsol.u...]

# l, times, states = loss(x0, opt_params);
# plot_fenrir!(ax12, (t=times, u=states), (t=tsteps, u=noisy_ode_data))

f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(
    f, x0, opt_params;
    lb=[-100.0, -100.0, 0.0, log(1e-8), log(1e-20)],
    ub=[100.0, 100.0, 100.0, log(1e4), log(1e50)])
# optimizer = LBFGS()
optsol = solve(optprob,
               # LBFGS(linesearch=Optim.LineSearches.BackTracking());
               LBFGS();
               maxiters=100, cb=plot_callback())
@info "DONE"

l, times, states = loss(optsol.u, opt_params);
# plot_fenrir!(ax13, (t=times, u=states), (t=tsteps, u=noisy_ode_data))
# push!(trajectories, (p0=p0, trajectory=copy(trajectory)))
