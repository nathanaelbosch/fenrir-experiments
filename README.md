# Fenrir: Physics-Enhanced Regression for Initial Value Problems - Experiments

This repo contains the experiment code for the paper "Fenrir: Physics-Enhanced Regression for Initial Value Problems", accepted at ICML 2022, currently available on [arXiv](https://arxiv.org/abs/2202.01287).


---

__To solve differential equations in Julia with probabilistic numerical solvers, have a look at our package
[ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl).__
Much of the new functionality presented in this paper is already available in ProbNumDiffEq.jl and will be covered in the documentation.

---

A __Python__ implementation of probabilistic numerical ODE solvers, as well as many other probabilistic numerical methods, is maintained in __[ProbNum](https://github.com/probabilistic-numerics/probnum)__ (`pip install probnum`).

---


## Running the experiments
The experiments are located in `./experiments/`; each subfolder has an individual README that explains the individual files.
Since the scripts use the core located in `./src/`, code should be run from the root directory directly.

First open `julia`, activate the local environment, and instantiate it to install all the packages:
```
julia> ]
(v1.7) pkg> activate .
(v1.7) pkg> instantiate
```
and you can quit the `pkg` environment by hitting backspace.

To run a julia script from the Julia REPL, say `./experiments/2/plot.jl`, do
```
julia> include("experiments/2/plot.jl")
```


## Reference
```
@misc{https://doi.org/10.48550/arxiv.2202.01287,
  title = {Fenrir: Physics-Enhanced Regression for Initial Value Problems},
  author = {Tronarp, Filip and Bosch, Nathanael and Hennig, Philipp},
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2202.01287},
  year = {2022},
  url = {https://arxiv.org/abs/2202.01287},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
