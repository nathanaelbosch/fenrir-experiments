#=
I think this script was meant to showcase that RK fails for high-frequency data
=#
using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using UnPack
using Random

prob, (tsteps, ode_data), noise_var, θ_init, θ_bounds, u0_bounds = pendulum()
proj = [0 1]
ode_data = [proj*u for u in ode_data]
true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10);
D = length(prob.u0)
RNG = MersenneTwister(1234)
noisy_ode_data = [u + sqrt.(noise_var) .* randn(RNG, size(u)) for u in ode_data];

Plots.plot(true_sol, color=:black, linestyle=:dash, label="")
Plots.scatter!(tsteps, ProbNumDiffEq.stack(noisy_ode_data), color=(1:D)', label="")

plot_callback = function ()
    j = 0
    function (θ, l, sol)
        j += 1

        u0 = θ[1:D]
        p = θ[D+1:end]

        # @info "[$i] [Iteration $j] Callback" u0 p l
        @info "[Iteration $j] Callback" u0 p l

        plt = Plots.scatter(tsteps, ProbNumDiffEq.stack(noisy_ode_data), label="", color=(1:D)',
                      markersize=2, markerstrokewidth=0.1)
        Plots.plot!(true_sol, color=:black, linestyle=:dot, label="")
        Plots.plot!(sol, label="", color=[2 1])
        Plots.plot!(ylims=(-2, 2))

        display(plt)

        return false
    end
end

u0 = [0.0, noisy_ode_data[1][1]]
u0 = clamp.(u0, u0_bounds[1], u0_bounds[2])
p0 = θ_init()
p0 = 5.0
θ_start = [u0..., p0...]


function loss(x, _)
    u0, p = x[1:D], x[D+1]
    sol = solve(remake(prob, u0=u0, p=p), Tsit5(), abstol=1e-6, reltol=1e-6, saveat=tsteps)
    if sol.retcode != :Success
        return 1e10, sol
    end

    # return sum(norm.([proj * u for u in sol.u] - noisy_ode_data) .^ 2), sol
    return mean(norm.([proj * u for u in sol.u] - noisy_ode_data)), sol
end
l, sol = loss(θ_start, nothing);
plot_callback()(θ_start, l, sol);
plot_classic!(ax21, sol, (t=tsteps, u=noisy_ode_data))


f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(
    f, θ_start, nothing;
    lb=[u0_bounds[1]..., θ_bounds[1]...],
    ub=[u0_bounds[2]..., θ_bounds[2]...]
)

# optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())
optimizer = LBFGS()

optsol = solve(optprob, optimizer; maxiters=100, cb=plot_callback())
_l, final_sol = loss(optsol.u, nothing)
plot_classic!(ax23, final_sol, (t=tsteps, u=noisy_ode_data))
