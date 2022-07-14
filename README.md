# Fenrir: Physics-Enhanced Regression for Initial Value Problems - Experiments

This repo contains the experiment code for the paper "Fenrir: Physics-Enhanced Regression for Initial Value Problems", accepted at ICML 2022 ([link](https://proceedings.mlr.press/v162/tronarp22a.html])).

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
([link](https://proceedings.mlr.press/v162/tronarp22a.html))
```
@InProceedings{pmlr-v162-tronarp22a,
  title = 	 {Fenrir: Physics-Enhanced Regression for Initial Value Problems},
  author =       {Tronarp, Filip and Bosch, Nathanael and Hennig, Philipp},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {21776--21794},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/tronarp22a/tronarp22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/tronarp22a.html}
}
```
