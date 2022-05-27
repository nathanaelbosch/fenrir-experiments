# Experiment 4: High frequencies and large amounts of data

For figure 6, run:
- `./6_make_trajectory_plot.jl`; this calls `./6_run_rk.jl` and `./6_run_fenrir.jl`.

For figure 7, run:
- `./7_generate_loss_landscape_data.jl`, which saves data to `./loss_heatmap.jld`
- `./7_compute_optimizer_trajectories.jl` to generate the trajectory shown in red
- `./7_plot_loss_landscape.jl`, which creates figure 7, taking data from `./loss_heatmap.jld` and `./trajectories.jld`

For figure 8, run:
- `./8_evaluate_rk.jl`
- `./8_evaluate_fenrir.jl`
- `./8_plot_eval.jl`
