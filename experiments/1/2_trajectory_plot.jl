using OrdinaryDiffEq, LinearAlgebra, Statistics
using CairoMakie
# using Plots
using Fenrir

# probname, noisestr = "lv", "high"
probname, noisestr = "fhn", "high"

include("../theme.jl")
WIDTH, HEIGHT = FULL_WIDTH, HALF_HEIGHT / 2
DIR = @__DIR__
RESULTS_DIR = joinpath(@__DIR__, "results")

METHODS = ("ODIN_withIV", "Classic_withIV", "FENRIR_withIV")
NOISE_LEVELS = ("low", "high")

PROBLEMS = Dict(
    "lv" => lotkavolterra,
    "fhn" => fitzhughnagumo,
    "pt" => protein_transduction,
)
problem = PROBLEMS[probname]
prob, _, _, _, _, _ = problem()
D = length(prob.u0)
plot_ts = range(prob.tspan..., length=1000)
true_sol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-6, saveat=plot_ts)




get_params(filepath) = map(l -> parse.(Float64, l), split.(readlines(filepath), ","))
parameters = Dict([
    (probname, noisestr, m) => get_params(
        joinpath(RESULTS_DIR, "$(probname)_$(noisestr)noise_$(lowercase(m)).txt"))
    for m in METHODS, noisestr in NOISE_LEVELS])



# fig = Figure(
#     resolution=(WIDTH, HEIGHT),
#     figure_padding=5,
# )
axis_kwargs = (
    xgridvisible=false,
    ygridvisible=false,
    # yticks=[1,6],
    yticks=[-2,2],
    xticks=[prob.tspan[1], prob.tspan[2]],
    xtrimspine=true, ytrimspine=true,
)
# ax1 = fig[1, 1] = Axis(fig; axis_kwargs...)
# ax2 = fig[1, 2] = Axis(fig; axis_kwargs..., yticklabelsvisible=false)
# ax3 = fig[1, 3] = Axis(fig; axis_kwargs..., yticklabelsvisible=false)
# rowgap!(fig.layout, 15)
# colgap!(fig.layout, 15)
ax1 = Axis(gl[1,1]; axis_kwargs...,
           xticklabelsvisible=false,
           )
ax2 = Axis(gl[2,1]; axis_kwargs...,
           xticklabelsvisible=false,
           )
ax3 = Axis(gl[3,1]; axis_kwargs...,
           # yticklabelsvisible=false,
           )
axes = [ax1, ax2, ax3]



sums = []
for (i, m) in enumerate(METHODS)
    # if m == "FENRIR" continue end
    ps = parameters[(probname, noisestr, m)]
    eprob = EnsembleProblem(prob, prob_func=(prob, i, repeat)->remake(
        prob, u0=ps[i][1:D], p=ps[i][D+1:end]))
    sim = solve(eprob,
                # Tsit5(),
                RadauIIA5(),
                EnsembleThreads();
                trajectories=length(ps),
                abstol=1e-9, reltol=1e-6,
                )

    solutions = cat([ProbNumDiffEq.stack(s.(plot_ts)) for s in sim.u]..., dims=3)
    medians = median(solutions, dims=3)

    sum = EnsembleSummary(sim, plot_ts;
                          # quantiles=[0.05, 0.95]
                          quantiles=[0.10, 0.90]
                          )

    for d in 1:D
        # c = COLORS[D*(i-1)+d]
        c = COLORS[i]
        # for u in sim.u
        #     lines!(axes[i], u.t, [_u[d] for _u in u.u],
        #            color=(c, 1.0), linewidth=0.1)
        # end
        lines!(axes[i], sum.u.t,
               # [u[d] for u in sum.u.u],
               medians[:, d, 1],
               linewidth=1,
               color=(c, 1.0))
        band!(axes[i], sum.u.t,
              [u[d] for u in sum.qhigh.u],
              [u[d] for u in sum.qlow.u],
              color=(c, 0.4))
    end
end

for ax in (ax1, ax2, ax3)
    for d in 1:D
        lines!(ax, true_sol.t, Array(true_sol)[d, :],
               color=:black, linestyle=:dash, linewidth=0.5,
               )
    end
end

rowgap!(gl, 10)

# save(joinpath(DIR, "results", "trajectories.pdf"), fig, pt_per_unit=1)
