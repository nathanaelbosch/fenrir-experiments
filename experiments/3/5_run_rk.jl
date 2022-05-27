using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using ForwardDiff
using Fenrir
using UnPack

logistic(x) = @. 1 / (1 + exp(-x))
logistic_inverse(x) = @. log(x / (1 - x))

RESULTS_DIR = joinpath(@__DIR__, "results")

filename = "seir_classic.csv"
filepath = joinpath(RESULTS_DIR, filename)
run(`rm -f $filepath`)
@info "save into" filepath

prob = seir()
prob = ProbNumDiffEq.remake_prob_with_jac(prob)
true_sol = solve(prob, Tsit5(), abstol = 1e-10, reltol = 1e-10)

tsteps = 30:1:prob.tspan[2]
ode_data = true_sol.(tsteps)
proj = [0 0 1 0; 0 0 0 1]
ode_data = [proj * u for u in ode_data]
noise_var = 5e-4

const D = length(prob.u0)
const P = length(prob.p)
const E = length(ode_data[1])


for i = 1:100

    noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data]

    function x_to_theta(x)
        e0 = x[1]
        i0 = x[2]
        u0 = [1.0 - e0 - i0, e0, i0, zero(i0)]
        p = x[3:5]
        return (u0 = u0, p = p)
    end
    function loss(x, _)
        @unpack u0, p = x_to_theta(x)
        sol = solve(
            remake(prob, u0 = u0, p = p),
            Tsit5(),
            abstol = 1e-8,
            reltol = 1e-6,
            saveat = tsteps,
        )

        return sum(norm.([proj * u for u in sol.u] - noisy_ode_data) .^ 2), sol
        # return mean(norm.([proj * u for u in sol.u] - noisy_ode_data)), sol
    end

    e0 = abs(1e-2 * randn())
    i0 = abs(1e-2 * randn())
    p0 = [rand(), rand(), rand()]
    # x0 = [logistic_inverse(i0), logistic_inverse.(p0)...]
    x0 = [e0, i0, p0...]

    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    p_bd = i0_bd = [0.0, 1.0]
    optprob = OptimizationProblem(
        f,
        x0;
        lb = [0.0, 0.0, 0.0, 0.0, 0.0],
        ub = [1.0, 1.0, 1.0, 1.0, 1.0],
    )

    @info "[$i] Optimize"
    optimizer = LBFGS(linesearch = Optim.LineSearches.BackTracking())
    # optimizer = LBFGS()
    # optimizer = BFGS(initial_stepnorm=0.01);
    optsol = solve(optprob, optimizer; maxiters = 1000)
    # i0, p = logistic(optsol.u[1]), logistic.(optsol.u[2:end])
    @unpack u0, p = x_to_theta(optsol.u)
    @info "[$i] Done" loss = optsol.minimum u0_inferred = u0 u0_true = prob.u0 p_inferred =
        p p_true = prob.p

    open(filepath, "a") do f
        write(f, join(string.([u0..., p...]), ",") * "\n")
    end
end
