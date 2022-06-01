using LinearAlgebra, Statistics, Random
using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, Flux
using UnPack, CSV, DataFrames
using Fenrir

RESULTS_DIR = joinpath(@__DIR__, "results")

prob, (tsteps, ode_data), noise_var, θ_init, θ_bounds, u0_bounds = pendulum()
proj = [0 1]
ode_data = [proj * u for u in ode_data]
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
        p = θ[1]
        @info "[Iteration $j] Callback" p l
        plt = Plots.scatter(
            tsteps,
            ProbNumDiffEq.stack(noisy_ode_data),
            label="",
            color=(1:D)',
            markersize=2,
            markerstrokewidth=0.1,
        )
        Plots.plot!(true_sol, color=:black, linestyle=:dot, label="")
        Plots.plot!(sol, label="", color=[2 1])
        Plots.plot!(ylims=(-2, 2))
        display(plt)
        return false
    end
end

function loss(p::Real)
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps)
    if sol.retcode != :Success
        return 1e10, sol
    end
    return sum(norm.([proj * u for u in sol.u] - noisy_ode_data) .^ 2), sol
end
loss(x::AbstractVector, _) = loss(x[1])

# p0 = θ_init()
p0 = 5.0
x0 = [p0...]
l, sol = loss(x0[1]);
# plot_callback()(x0, l, sol);

function get_opt(p0; plotcb=true)
    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(f, [p0], nothing; lb=[0.0], ub=[100.0])
    optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())
    optsol = solve(
        optprob,
        optimizer;
        maxiters=2000,
        cb=plotcb ? plot_callback() : (args...) -> false,
    )
    return optsol.u[1]
end
# get_opt(p0)

df = DataFrame(p0=Float64[], L0=Float64[], popt=Float64[], Lopt=Float64[])
for p0 in 10.0 .^ (-2:0.005:2)
    @info "Get Loss" p0
    L0 = loss(p0)[1]
    push!(df, (p0, L0, NaN, NaN))
end
for p0 in 10.0 .^ (-2:0.025:2)
    @info "Get popt" p0
    popt = get_opt(p0; plotcb=false)
    Lopt = loss(popt)[1]
    (@view df[df.p0.==p0, :]).popt .= popt
    (@view df[df.p0.==p0, :]).Lopt .= Lopt
end

p1 = plot(df.p0, df.L0, xscale=:log10, xlabel="p0", ylabel="L0")
p2 = scatter(df.p0, df.popt, xscale=:log10, yscale=:log10, xlabel="p0", ylabel="popt")
hline!([1], color=:black, linestyle=:dash)
plot(p1, p2, layout=(2, 1), legend=false)

CSV.write(joinpath(RESULTS_DIR, "rk_df.csv"), df)
