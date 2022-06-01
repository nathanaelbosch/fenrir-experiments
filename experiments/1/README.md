# Experiment 1: Benchmarking Estimation Accracy

We compare Fenrir to ODIN and to a Runge-Kutta based inference method, on the Lotka-Volterra and FitzHugh-Nagumo problem; each with 2 noise levels.

The scripts to compute the ODIN results are
- `./run_odin_lv.py`
- `./run_odin_fhn.py`
To run the scripts one needs to either run them inside the ODIN source directory, or install ODIN (by creating a small setup.py) file.
If you encounter issues, make sure that you install ODIN with Tensorflow 1.14.

The other scripts that need to be run are:
- `./run_rk.jl` to compute the RK results
- `./run_fenrir.jl` to compute the Fenrir results
- `./make_figure2.jl` to compute the final plot (calls `./_trajectory_plot.jl`)

`./make_figure9.jl` creates Figure 9 that can be found in the appendix.
