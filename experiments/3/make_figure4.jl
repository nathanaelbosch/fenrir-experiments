using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, LinearAlgebra, Statistics
using ForwardDiff
using Fenrir
using UnPack
using CairoMakie
using ColorSchemes

DIR = @__DIR__

prob = Fenrir.seir()
prob = ProbNumDiffEq.remake_prob_with_jac(prob)
true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)

tsteps = 30:1:prob.tspan[2]
ode_data = true_sol.(tsteps)
proj = [0 0 1 0; 0 0 0 1]
ode_data = [proj * u for u in ode_data]
noise_var = 5e-4
noisy_ode_data = [clamp.(u + sqrt.(noise_var) .* randn(size(u)), 0, 1) for u in ode_data]

const D = length(prob.u0)
const P = length(prob.p)
const E = length(ode_data[1])

function vec_to_params(x)
    e0 = x[1]
    i0 = x[2]
    u0 = [1.0 - e0 - i0, e0, i0, zero(i0)]
    p = x[3:5]
    return (u0=u0, p=p, σ²=exp(x[6]), κ²=exp(x[7]))
end
function loss(x, other_params)
    @unpack u0, p, σ², κ² = vec_to_params(x)
    @unpack prob, ode_data, tsteps, proj, dt = other_params
    data = (t=tsteps, u=noisy_ode_data)
    return exact_nll(remake(prob, u0=u0, p=p), data, σ², κ², ; dt=dt, proj=proj)
end

κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps, proj)
e0 = abs(1e-2 * randn())
i0 = abs(1e-2 * randn())
e0 = 0.001
i0 = 0.0001
p0 = [rand(), rand(), rand()]
p0 = [0.9, 0.1, 0.5]
x0 = [e0, i0, p0..., log(1.0), log(κ²0)]

opt_params = (prob=prob, ode_data=noisy_ode_data, tsteps=tsteps, dt=0.2, proj=proj);
loss(x0, opt_params)

f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(
    f,
    x0,
    opt_params;
    lb=[0.0, 0.0, 0.0, 0.0, 0.0, log(1e-10), log(1e-20)],
    ub=[1.0, 1.0, 1.0, 1.0, 1.0, log(1e2), log(1e20)],
)

optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())
_cb() = (
    j = 1;
    (x, l, args...) -> begin
        j += 1
        @info "Iteration $j" loss = l e0 = x[1] e0_true = prob.u0[2] i0 = x[2] i0_true =
            prob.u0[3] p = x[3:3+P-1] p_true = prob.p
        false
    end
)
optsol = solve(optprob, optimizer; maxiters=1000, cb=_cb())

include("../theme.jl")

COLORS = ColorSchemes.tab10.colors
fig = Figure(resolution=(HALF_WIDTH, HALF_HEIGHT), figure_padding=5)
ax1 =
    fig[1, 1] = Axis(
        fig;
        xticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        xticks=[0, 100],
        yticks=[0, 1],
        xtrimspine=true,
        ytrimspine=true,
    )
ax2 =
    fig[2, 1] = Axis(
        fig;
        xgridvisible=false,
        ygridvisible=false,
        xticks=[0, 100],
        yticks=[0, 1],
        xtrimspine=true,
        ytrimspine=true,
    )
for (x, l, j) in [(x0, "initial", 1), (optsol.u, "final", 2)]
    NLL, times, states = loss(x, opt_params)

    for i in 1:4
        means = [u.μ[i] for u in states]
        stds = [u.Σ[i, i] for u in states]
        lines!(fig[j, 1], times, means, label="SEIR"[i], color=COLORS[i])
        band!(ax2, times, means - 2stds, means + 2stds; color=(COLORS[i], 0.2))
    end
    for i in 1:2
        Makie.scatter!(
            fig[j, 1],
            tsteps,
            [u[i] for u in noisy_ode_data],
            color=COLORS[[3, 4][i]],
            markersize=2,
        )
    end
end
labels = ["S", "E", "I", "R"]
elements = [PolyElement(polycolor=COLORS[i]) for i in 1:length(labels)]
Legend(
    fig[:, 2],
    elements,
    labels,
    "",
    margin=(0, 0, 0, 0),
    padding=(0, 0, 0, 0),
    position=:rt,
)
rowgap!(fig.layout, 10)
colgap!(fig.layout, 10)
save(joinpath(DIR, "figure4.pdf"), fig, pt_per_unit=1)
