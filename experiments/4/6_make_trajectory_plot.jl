using CairoMakie

include("../theme.jl")


fig = Figure(
    resolution=(FULL_WIDTH, FULL_HEIGHT),
    # resolution=(HALF_WIDTH, HALF_HEIGHT),
    figure_padding=5,
)

axis_kwargs = (
    xgridvisible=false,
    ygridvisible=false,
    yticks=[-2,2],
    xticks=[0,10],
    xtrimspine=true, ytrimspine=true,
)
# g1 = fig[1, 1] = GridLayout()
# g2 = fig[1, 2] = GridLayout()
ax11 = fig[2, 1] = Axis(fig; axis_kwargs...)
ax12 = fig[2, 2] = Axis(fig; axis_kwargs...)
ax13 = fig[2, 3] = Axis(fig; axis_kwargs...)
ax21 = fig[1, 1] = Axis(fig; axis_kwargs...)
# ax22 = fig[2, 2] = Axis(fig; axis_kwargs...)
ax23 = fig[1, 2] = Axis(fig; axis_kwargs...)
# ax11 = Axis(g2[1, 1]; xticklabelsvisible=false, axis_kwargs...)
# ax12 = Axis(g2[2, 1]; xticklabelsvisible=false, axis_kwargs...)
# ax13 = Axis(g2[3, 1]; axis_kwargs...)
# ax21 = Axis(g1[1, 1]; xticklabelsvisible=false, axis_kwargs...)
# ax22 = Axis(g1[1, 1]; axis_kwargs...)
# ax23 = Axis(g1[2, 1]; axis_kwargs...)
for ax in (ax11, ax12, ax13, ax21, ax23)
    Makie.xlims!(ax, -0.5, 10.5)
    Makie.ylims!(ax, -2.5, 2.5)
end

update_theme!(
    Lines = (
        linewidth=1,
    ),
    Scatter = (
        markersize=1,
        markerstrokewidth=0.1,
    ),
)

plot_data!(ax, data) = Makie.scatter!(
    ax, data.t, [u[1] for u in data.u],
    label="",
    # color=1,
    color=(:black, 0.8),
    markersize=1, markerstrokewidth=0.1
)

function plot_classic!(ax, sol, data)
    plot_data!(ax, data)
    for d in 1:2
        lines!(
            ax, sol.t, [u[d] for u in sol.u],
            color=(COLORS[2], 0.8),
            linestyle= d==2 ? :solid : :dash,
        )
    end
end


function plot_fenrir!(ax, sol, data)
    plot_data!(ax, data)
    for d in 1:2
        means = [u.μ[d] for u in sol.u]
        stds = [sqrt(u.Σ[d,d]) for u in sol.u]
        lines!(
            ax, sol.t, means,
            color=(COLORS[3], 0.8),
            linestyle= d==2 ? :solid : :dash,
        )
        band!(
            ax, sol.t, means-2stds, means+2stds,
            color=(COLORS[3], 0.2),
        )
    end
end


include("run_classic.jl")
include("run_fenrir.jl")



labels = [
    "RK",
    "FENRIR",
]
elements = [PolyElement(polycolor = COLORS[[2,3][i]]) for i in 1:length(labels)]
labels2 = ["Position", "Velocity", "Data"]
elements2 = [
    LineElement(color=:black, linewidth=1,),
    LineElement(color=:black, linestyle=:dash, linewidth=1),
    MarkerElement(marker=:cicle, color=:black, markersize=3),
]
Legend(fig[1, 3],
       [elements, elements2],
       [labels, labels2],
       [nothing, nothing],
       margin=(0,0,0,0),
       padding=(0,0,0,0),
       tellheight=false, tellwidth=false,
       orientation=:horizontal,
       nbanks=3,
       )

rowgap!(fig.layout, 10)
colgap!(fig.layout, 5)
# rowgap!(g1, 5)
# rowgap!(g2, 5)

label_kwargs = (
    padding=(0, 5, 5, 0),
    halign=:right,
    # halign=:left,
    textsize=BASE_FONTSIZE-3,
    tellwidth=false,
    tellheight=false,
)
# labels = [
#     "A: RK initialization",
#     "B: RK optimum",
#     "C: Fenrir initialization",
#     "D: Fenrir with increased diffusion",
#     "E: Fenrir optimum",
# ]
labels = ["A","B","C","D","E"]
Label(fig[1,1,TopLeft()], labels[1]; label_kwargs)
Label(fig[1,2,TopLeft()], labels[2]; label_kwargs)
Label(fig[2,1,TopLeft()], labels[3]; label_kwargs)
Label(fig[2,2,TopLeft()], labels[4]; label_kwargs)
Label(fig[2,3,TopLeft()], labels[5]; label_kwargs)

save("experiments/4/trajectory_plot.pdf", fig, pt_per_unit=1)
