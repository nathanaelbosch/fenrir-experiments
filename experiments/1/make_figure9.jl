using OrdinaryDiffEq, LinearAlgebra, Statistics
using CairoMakie
using Fenrir
using ProbNumDiffEq

include("../theme.jl")
DIR = @__DIR__

WIDTH, HEIGHT = FULL_WIDTH, HALF_HEIGHT * 0.9

RESULTS_DIR = joinpath(@__DIR__, "results")
RESULTS_DIR = "experiments/1/results"

PROBLEMS = Dict("lv" => lotkavolterra, "fhn" => fitzhughnagumo)
NOISE_LEVELS = ("low", "high")
METHODS = ("ODIN_withIV", "Classic_withIV", "FENRIR_withIV")
LABELS = Dict(
    "lv" => ["y₁(0)", "y₂(0)", "α", "β", "γ", "δ"],
    "fhn" => ["y₁(0)", "y₂(0)", "a", "b", "c"],
)

RELATIVE_ERROR = false

subplot = 'a'
for probname in keys(PROBLEMS), noisestr in NOISE_LEVELS
    @info "Plotting" probname noisestr
    problem = PROBLEMS[probname]

    fig = Figure(resolution=(WIDTH, HEIGHT), figure_padding=5)
    ax =
        fig[1, 1] = Axis(
            fig;
            ylabel=RELATIVE_ERROR ? "Relative error" : "Absolute error",
            yscale=log10,
            xgridvisible=false,
            ygridvisible=true,
            bottomspinevisible=false,
            leftspinevisible=false,
            ygridcolor=:black,
            ygridwidth=0.5,
            yticksvisible=false,
            xticksvisible=false,
            # xticks=[1e-2, 1e-1, 1e0],
            xticklabelsize=BASE_FONTSIZE - 1,
            yticks=(
                [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²"],
            ),
        )

    prob, (tsteps, _), _, _, _ = problem()
    true_theta = [prob.u0..., prob.p...]
    D = length(prob.u0)
    P = length(prob.p)

    get_params(filepath) = map(l -> parse.(Float64, l), split.(readlines(filepath), ","))
    parameters = Dict([
        (m) => get_params(
            joinpath(RESULTS_DIR, "$(probname)_$(noisestr)noise_$(lowercase(m)).txt"),
        ) for m in METHODS
    ])
    params = [parameters[m] for m in METHODS]
    N1, N2, N3 = length.([parameters[m] for m in METHODS])
    all_params = vcat(ProbNumDiffEq.stack.([parameters[m] for m in METHODS])...)

    abs_errors = abs.(all_params .- true_theta')
    rel_errors = abs_errors ./ abs.(true_theta')
    errors = RELATIVE_ERROR ? rel_errors : abs_errors

    for j in 1:length(true_theta)
        ys = errors[:, j]
        xs = j * ones(Int, length(ys))
        dodge = vcat(ones(Int, N1), 2 * ones(Int, N2), 3 * ones(Int, N3))

        boxplot!(
            ax,
            xs,
            ys;
            dodge=dodge,
            color=[COLORS[d] for d in dodge],
            BOXPLOT_KWARGS...,
        )
    end

    labels = ["ODIN", "RK", "FENRIR"]
    elements = [PolyElement(polycolor=COLORS[i]) for i in 1:length(labels)]
    title = "Method"
    Legend(fig[1, end+1], elements, labels, title)

    labels = LABELS[probname]
    ax.xticks = (1:length(labels), labels)

    trim!(fig.layout)

    # save(joinpath(DIR, "results", "p_errs_$(probname)_$(noisestr).pdf"), fig, pt_per_unit=1)
    save(joinpath(DIR, "figure9$(subplot).pdf"), fig, pt_per_unit=1)
    subplot += 1
end
