module Fenrir

using LinearAlgebra
using Statistics
using Logging
using Distributions: logpdf
using ProbNumDiffEq
using ProbNumDiffEq: X_A_Xt
using OrdinaryDiffEq
using DifferentialEquations
using UnPack
using ForwardDiff

include("odin_problems.jl")
export lotkavolterra, fitzhughnagumo, protein_transduction
include("other_problems.jl")
export hodgkinhuxley, sird, seir, pendulum, logistic

include("exact_likelihood.jl")
export exact_nll, get_initial_diff

end
