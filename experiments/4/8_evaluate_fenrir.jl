#=
I think this script was meant to showcase that RK fails for high-frequency data
=#
using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics, Flux
using Fenrir
using UnPack
using Random
using DataFrames

prob, (tsteps, ode_data), noise_var, θ_init, θ_bounds, u0_bounds = pendulum()
proj = [0 1]
ode_data = [proj*u for u in ode_data]
true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10);
D = length(prob.u0)
RNG = MersenneTwister(1234)
noisy_ode_data = [u + sqrt.(noise_var) .* randn(RNG, size(u)) for u in ode_data];

Plots.plot(true_sol, color=:black, linestyle=:dash, label="")
Plots.scatter!(tsteps, ProbNumDiffEq.stack(noisy_ode_data), color=(1:D)', label="")

callback = function ()
    j = 0
    function (x, l, args...)
        j += 1
        @info "[Iteration $j] Callback" l p=x[1] σ²=exp(x[end-1]) κ²=exp(x[end])
        return false
    end
end

function loss(x, other_params)
    p, σ², κ² = x[1], exp(x[end-1]), exp(x[end])
    @unpack prob, ode_data, tsteps, dt, proj = other_params
    data = (t=tsteps, u=noisy_ode_data)
    return exact_nll(remake(prob, p=p), data, σ², κ²; dt=dt, proj=proj)
end

# p0 = θ_init()
p0 = 5.0
κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps, proj)
σ²0 = 1.0
x0 = [p0, log(σ²0), log(κ²0)]

opt_params = (
    prob=prob,
    ode_data=noisy_ode_data,
    tsteps=tsteps,
    # dt=1e-1,
    dt=false,
    proj=proj,
);

l, times, states = loss(x0, opt_params);

diff_loss(x, other_params) = loss([x0[1:end-2]..., x...], other_params)
function get_opt(p0)
    optimizer = LBFGS()
    diff_f = OptimizationFunction(diff_loss, GalacticOptim.AutoForwardDiff())
    diff_optprob = OptimizationProblem(
        diff_f, [log(σ²0), log(κ²0)], opt_params;
        lb=[log(1e-8), log(1e-20)],
        ub=[log(1e4), log(1e50)]
    )
    optsol = solve(diff_optprob, optimizer; maxiters=200, cb=callback())

    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(
        f,
        # [p0, log(σ²0), log(κ²0)],
        [p0, optsol.u...],
        opt_params;
        lb=[0.0, log(1e-8), log(1e-20)],
        ub=[100.0, log(1e4), log(1e50)]
    )
    # optimizer = LBFGS()
    optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())
    optsol = solve(optprob, optimizer; maxiters=200, cb=callback())
    return optsol.u[1]
end
get_opt(p0)

prange = 10.0 .^ (-2:0.025:2)
df = DataFrame(p0=Float64[], L0=Float64[], popt=Float64[], Lopt=Float64[])
_loss(p0) = loss([p0, log(σ²0), log(κ²0)], opt_params)[1]
for p0 in prange
    @info "" p0
    L0 = _loss(p0)
    popt = get_opt(p0)
    Lopt = _loss(popt)
    push!(df, (p0, L0, popt, Lopt))
end

# p1 = plot(df.p0, df.L0, xscale=:log10, xlabel="p0", ylabel="L0")
# p2 = scatter(df.p0, df.popt, xscale=:log10, yscale=:log10, xlabel="p0", ylabel="popt")
# hline!([1], color=:black, linestyle=:dash)
# plot(p1, p2, layout=(2,1), legend=false)

CSV.write("experiments/4/fenrir_df.csv", df)
