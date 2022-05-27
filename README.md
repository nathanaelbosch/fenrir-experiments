# Fenrir: Physics-Enhanced Regression for Initial Value Problems - Experiments

This repo contains the experiment code for the paper "Fenrir: Physics-Enhanced Regression for Initial Value Problems", accepted at ICML 2022, currently available on [arXiv](https://arxiv.org/abs/2202.01287).


---

__The functionality of this paper is available in [Fenrir.jl](https://github.com/nathanaelbosch/Fenrir.jl).__
If you want to use the new PN marginal likelihood for an inference problem, we recommend to use Fenrir.jl and not this repo.

---

__If you want to solve differential equations with probabilistic numerical solvers, have a look at [ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl).__
It contains _fast_ ODE filters for first- and second-order ODEs, and even DAEs.
And if you prefer Python, just use one of the many solvers implemented in __[ProbNum](https://github.com/probabilistic-numerics/probnum)__ (`pip install probnum`).

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
