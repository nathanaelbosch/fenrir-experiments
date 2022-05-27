using OrdinaryDiffEq, LinearAlgebra, Statistics
using CairoMakie
using Fenrir
using CSV, DataFrames

DIR = @__DIR__
RESULTS_DIR = joinpath(DIR, "results")

include(joinpath(DIR, "../theme.jl"))
WIDTH, HEIGHT = HALF_WIDTH, HALF_HEIGHT

TRUE_PROB = Fenrir.seir()
TRUE_SOLUTION = solve(TRUE_PROB, Vern9(), abstol = 1e-9, reltol = 1e-9)
TIMES = 0:1:100
us_true = TRUE_SOLUTION.(TIMES)


RELATIVE_ERROR = false
fig = Figure(resolution = (WIDTH, HEIGHT), figure_padding = 5)
ax =
    fig[1, 1] = Axis(
        fig;
        yscale = log10,
        xgridvisible = false,
        ygridvisible = true,
        bottomspinevisible = false,
        leftspinevisible = false,
        ygridcolor = :black,
        ygridwidth = 0.5,
        yticksvisible = false,
        xticksvisible = false,
        ylabel = RELATIVE_ERROR ? "Relative error" : "Absolute error",
        xticks = ([1, 2, 3, 4, 5], [L"E_0", L"I_0", L"\beta", L"\gamma", L"\eta"]),
        xticklabelsize = BASE_FONTSIZE - 1,
    )
i0_rmses = Float64[]
p1_rmses = Float64[]
p2_rmses = Float64[]
p3_rmses = Float64[]
prmses = Float64[]
sizes = []
for c in ("classic", "fenrir")
    df = CSV.read(joinpath(RESULTS_DIR, "seir_$(c).csv"), DataFrame, header = false)
    @assert nrow(df) == 100
    push!(sizes, nrow(df))
    e0 = abs.(df[:, 2] .- TRUE_PROB.u0[2])
    i0 = abs.(df[:, 3] .- TRUE_PROB.u0[3])
    p1 = abs.(df[:, 5] .- TRUE_PROB.p[1])
    p2 = abs.(df[:, 6] .- TRUE_PROB.p[2])
    p3 = abs.(df[:, 7] .- TRUE_PROB.p[3])
    if RELATIVE_ERROR
        e0 ./= TRUE_PROB.u0[2]
        i0 ./= TRUE_PROB.u0[3]
        p1 ./= TRUE_PROB.p[1]
        p2 ./= TRUE_PROB.p[2]
        p3 ./= TRUE_PROB.p[3]
    end

    append!(prmses, vcat(e0, i0, p1, p2, p3))
end


O1, O2 = ones.(Int, sizes)
xs = vcat(vcat(O1, 2O1, 3O1, 4O1, 5O1), vcat(O2, 2O2, 3O2, 4O2, 5O2))
dodge = vcat(repeat(O1, 5), repeat(2O2, 5))
boxplot!(
    ax,
    xs,
    prmses;
    dodge = dodge,
    color = [COLORS[d+1] for d in dodge],
    BOXPLOT_KWARGS...,
)


labels = ["RK", "FENRIR"]
elements = [PolyElement(polycolor = COLORS[i+1]) for i = 1:length(labels)]
title = "Method"
axislegend(ax, elements, labels, "", position = :lt, margin = (5, 0, 0, 0))

save(joinpath(RESULTS_DIR, "seir_parameter_errors.pdf"), fig, pt_per_unit = 1)
