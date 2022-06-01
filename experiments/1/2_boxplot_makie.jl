using OrdinaryDiffEq, LinearAlgebra, Statistics
using CairoMakie
using Fenrir

include("../theme.jl")
DIR = @__DIR__

WIDTH, HEIGHT = FULL_WIDTH, HALF_HEIGHT

RESULTS_DIR = joinpath(@__DIR__, "results")
RESULTS_DIR = "experiments/1/results"

PROBLEMS = ((lotkavolterra, "lv"), (fitzhughnagumo, "fhn"))

NOISE_LEVELS = ("low", "high")

METHODS = ("ODIN_withIV", "Classic_withIV", "FENRIR_withIV")

fig = Figure(resolution=(WIDTH, HEIGHT), figure_padding=5)
gl = fig[1, 1] = GridLayout()
ax =
    fig[1, 2] = Axis(
        fig;
        ylabel="tRMSE",
        yscale=log10,
        xgridvisible=false,
        ygridvisible=true,
        bottomspinevisible=false,
        leftspinevisible=false,
        ygridcolor=:black,
        ygridwidth=0.5,
        yticksvisible=false,
        xticksvisible=false,
        xticks=[1e-2, 1e-1, 1e0],
        # xticklabelsize=BASE_FONTSIZE-1,
        yticks=(
            [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²"],
        ),
    )

for (j, (problem, probname)) in enumerate(PROBLEMS)
    @info "Problem" probname

    prob, (tsteps, _), _, _, _ = problem()

    true_sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=tsteps)
    D = length(true_sol.u[1])
    P = length(prob.p)
    function trajectory_rmse(prob, p, tsteps)
        if length(p) > P
            _u0 = p[1:D]
            _p = p[D+1:end]
            sol = solve(
                remake(prob, u0=_u0, p=_p),
                RadauIIA5(),
                abstol=1e-10,
                reltol=1e-10,
                saveat=tsteps,
                maxiters=1e7,
            )
            return sqrt(mean(norm.(sol.u - true_sol.u) .^ 2))
        else
            sol = solve(
                prob,
                RadauIIA5(),
                abstol=1e-10,
                reltol=1e-10,
                p=p,
                saveat=tsteps,
                maxiters=1e7,
            )
            return sqrt(mean(norm.(sol.u - true_sol.u) .^ 2))
        end
    end
    get_params(filepath) = map(l -> parse.(Float64, l), split.(readlines(filepath), ","))
    get_trmses(params) = map(p -> trajectory_rmse(prob, p, tsteps), params)
    get_prmses(params) =
        map(p -> length(p) > P ? norm(p[D+1:end] - prob.p) : norm(p - prob.p), params)

    parameters = Dict([
        (probname, noisestr, m) => get_params(
            joinpath(RESULTS_DIR, "$(probname)_$(noisestr)noise_$(lowercase(m)).txt"),
        ) for m in METHODS, noisestr in NOISE_LEVELS
    ])
    trmses = Dict([(@info "Compute tRMSEs" k; k => get_trmses(v)) for (k, v) in parameters])

    for (i, noisestr) in enumerate(NOISE_LEVELS)
        @info "Plot TRMSEs" probname noisestr

        asdf = [trmses[(probname, noisestr, m)] for m in METHODS]
        N1, N2, N3 = length.(asdf)
        _trmses = vcat(asdf...)
        _xs = (j - 1) * 2 .+ i * ones(Int, length(_trmses))
        _dodge = vcat(ones(Int, N1), 2 * ones(Int, N2), 3 * ones(Int, N3))
        labels = [METHODS[d] for d in _dodge]

        boxplot!(
            ax,
            _xs,
            _trmses;
            dodge=_dodge,
            color=[COLORS[d] for d in _dodge],
            BOXPLOT_KWARGS...,
        )
    end
end

labels = ["ODIN", "RK", "FENRIR"]
elements = [PolyElement(polycolor=COLORS[i]) for i in 1:length(labels)]
title = "Method"
Legend(fig[1, end+1], elements, labels, title)

ax.xticks = (
    [1, 2, 3, 4],
    [
        "Lotka-Volterra\n(low noise)",
        "Lotka-Volterra\n(high noise)",
        "FitzHugh-Nagumo\n(low noise)",
        "FitzHugh-Nagumo\n(high noise)",
    ],
)

trim!(fig.layout)

colsize!(fig.layout, 1, Auto(0.4))
colgap!(fig.layout, 10)
save(joinpath(DIR, "results", "trmses.pdf"), fig, pt_per_unit=1)
include(joinpath(DIR, "2_trajectory_plot.jl"))
save(joinpath(DIR, "results", "trmses.pdf"), fig, pt_per_unit=1)
