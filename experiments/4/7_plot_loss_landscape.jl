using Fenrir
using UnPack
using JLD
using Makie

# DIR = "experiments/4"
DIR = @__DIR__
d = load(joinpath(DIR, "loss_heatmap.jld"))
@unpack ps, losses, gradients, kappas_log10 = d

@unpack trajectories = load(joinpath(DIR, "trajectories.jld"))

include("../theme.jl")
WIDTH, HEIGHT = HALF_WIDTH, HALF_HEIGHT

fig = Figure(
    resolution=(HALF_WIDTH, HALF_HEIGHT),
    figure_padding=5)
ax = fig[1, 1] = Axis(
    fig;
    xlabel="Pendulum Length",
    ylabel="Diffusion",
    topspinevisible=true,
    rightspinevisible=true,
    yticks=([10, 20, 30], ["10¹⁰", "10²⁰", "10³⁰"]),
    # xscale=log10, xticks=([1e-1, 1e0, 1e1], ["10⁻¹", "10⁰", "10¹"]),
    xticks=[1, 5, 10],
)
xlims!(ax, minimum(ps), maximum(ps))
ylims!(ax, minimum(kappas_log10), maximum(kappas_log10))

co = contourf!(ax, ps, kappas_log10,
               losses,
               # levels=range(0, 15000, length=20),
               colormap=ColorSchemes.ice,
               )

Colorbar(fig[1, 2], co,
         label="Negative log-likelihood [10³]",
         size=5,
         # ticks=([0, 5000, 10000, 15000], ["0", "5000", "10000", "15000"]),
         ticks=([0, 5000, 10000, 15000], ["0", "5", "10", "15"]),
         # ticks=([0, 1], ["0", "1e-122"]),
         # ticklabelrotation=-π/4,
         )

colgap!(fig.layout, 10)


# Uncomment this to get the inset plot!
ax_in = Axis(fig, bbox=BBox(WIDTH*0.49, WIDTH*0.72, HEIGHT*0.62, HEIGHT*0.94),
             xticksvisible=false, yticksvisible=false,
             topspinevisible=true, rightspinevisible=true,
             )
probs = exp.(-losses)
probs ./= maximum(probs)
contourf!(ax_in, ps, kappas_log10, probs,
          # levels=range(0, 15000, length=20),
          # colormap=:dense,
          colormap=reverse(ColorSchemes.ice),
          )
text!(ax_in, "Likelihood", position=(3,26), textsize=BASE_FONTSIZE-2)




trajectories = [trajectories[3]]
for (i, (p0, trajectory)) in enumerate(trajectories)
# trajectory = trajectories[3][2]
    # trajectory = trajectory[7:end]
    xs, ys = [t.p for t in trajectory], log10.([t.diffusion for t in trajectory])

    sizes = 2 * ones(Int, length(xs))
    sizes[1] = 5
    sizes[end] = 5
    scatterlines!(ax, xs, ys,
                  color=([:red, :blue, :green, :orange][i], 0.8),
                  # color=(:red, 0.8),
                  # marker=:x,
                  # markersize=5,
                  # markersize=range(2, 1, length=length(xs)) .^ 2,
                  # markersize=3,
                  markersize=sizes,
                  strokewidth=0.2,
                  linewidth=1,
                  )

    text!(ax, "Initial\nparameter",
          position = (xs[1], ys[1]),
          align = (:left, :center),
          offset=(5,0),
          textsize=8,
          )
    text!(ax, "Optimized\nparameter",
          position = (xs[end], ys[end]),
          align = (:left, :center),
          offset=(5,0),
          textsize=7,
          )
end







save(joinpath(DIR, "loss_landscape.pdf"), fig, pt_per_unit=1)
