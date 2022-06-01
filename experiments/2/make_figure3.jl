using Printf, Statistics, CairoMakie

include("../theme.jl")
DIR = @__DIR__

COLORS = ColorSchemes.Pastel1_4.colors

filepath = joinpath(DIR, "results.txt")
lines = readlines(filepath)
split.(lines)
vals = map(lines) do (l)
    v = split(l, ",")
    name = v[1]
    vals = parse.(Float64, v[2:end])
    return vals
end

fig = Figure(resolution=(HALF_WIDTH, HALF_HEIGHT))
ax =
    fig[1, 1] = Axis(
        fig;
        ylabel="Negative log-likelihood",
        xgridvisible=false,
        ygridvisible=true,
        bottomspinevisible=false,
        leftspinevisible=false,
        ygridcolor=:black,
        ygridwidth=0.5,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsize=BASE_FONTSIZE - 1,
    )

for i in 1:4
    mval = mean(vals[i])
    mvalstr = mval < 10^5 ? @sprintf("%.2f", mval) : @sprintf("%.2e", mval)

    medval = median(vals[i])
    medvalstr = medval < 10^5 ? @sprintf("%.2f", medval) : @sprintf("%.2e", medval)

    @info "" i mvalstr medvalstr

    xs = repeat([i], length(vals[i]))
    boxplot!(ax, xs, vals[i]; color=COLORS[i], BOXPLOT_KWARGS...)
end

labels = ["M₁₁", "M₁₀", "M₀₁", "M₀₀"]
ax.xticks = ([1, 2, 3, 4], labels)
ylims!(ax, -100, 250)
ax.yticks = [-100, 0, 100, 200]
trim!(fig.layout)
save(joinpath(DIR, "figure3.pdf"), fig, pt_per_unit=1)
