# Experiment 4: High frequencies and large amounts of data

For figure 6, run:
- `./make_figure6.jl`; this calls `./_figure6_run_rk.jl` and `./_figure6_run_fenrir.jl`.

For figure 7, run
- `./pre_figure7_genlandscape.jl`, which saves data to `./results/loss_heatmap.jld`
- `./pre_figure7_gentraj.jl` to generate the trajectory shown in red
- `./make_figure7.jl`, which creates figure 7, taking data from `./loss_heatmap.jld` and `./trajectories.jld`

For figure 8, run:
- `./pre_figure8_evaluate_rk.jl`
- `./pre_figure8_evaluate_fenrir.jl`
- `./make_figure8.jl`
