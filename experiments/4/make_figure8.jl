using CSV, DataFrames, CairoMakie

include("../theme.jl")
DIR = @__DIR__

df_rk = CSV.read("experiments/4/results/rk_df.csv", DataFrame)
df_fenrir = CSV.read("experiments/4/results/fenrir_df.csv", DataFrame)

fig = Figure(resolution=(HALF_WIDTH, HALF_HEIGHT), figure_padding=5)
axis_kwargs = (xscale=log10, xticks=([1e-2, 1e0, 1e2], ["10⁻²", "10⁰", "10²"]))
ax =
    fig[1, 1] = Axis(
        fig;
        yscale=log10,
        xlabel="Initial parameter",
        ylabel="Inferred parameter",
        axis_kwargs...,
    )
ylims!(ax, 2e-1, 2e2)

sc1 = scatter!(
    ax,
    df_fenrir.p0,
    df_fenrir.popt,
    markersize=8,
    color=(COLORS[3], 0.8),
    marker=:circle,
)
sc2 = scatter!(
    ax,
    df_rk.p0,
    df_rk.popt,
    markersize=8,
    color=(COLORS[2], 0.8),
    marker=:diamond,
)
hl = hlines!(ax, [1], color=:black, linestyle=:dash, linewidth=1)

axislegend(ax, [sc2, sc1], ["RK", "FENRIR"], nothing, position=:lt)

rowgap!(fig.layout, 10)
save(joinpath(DIR, "figure8.pdf"), fig, pt_per_unit=1)
