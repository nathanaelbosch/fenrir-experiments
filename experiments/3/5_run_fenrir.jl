using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using ForwardDiff
using Fenrir
using UnPack

logistic(x) = @. 1 / (1 + exp(-x))
logistic_inverse(x) = @. log(x / (1 - x))

RESULTS_DIR = joinpath(@__DIR__, "results")

filename = "seir_fenrir.txt"
filepath = joinpath(RESULTS_DIR, filename)
run(`rm -f $filepath`)
@info "save into" filepath

prob = seir()
prob = ProbNumDiffEq.remake_prob_with_jac(prob)
true_sol = solve(prob, Tsit5(), abstol = 1e-10, reltol = 1e-10)

tsteps = 30.0:1.0:prob.tspan[2]
ode_data = true_sol.(tsteps)
proj = [0 0 1 0; 0 0 0 1]
ode_data = [proj * u for u in ode_data]
noise_var = 5e-4

const D = length(prob.u0)
const P = length(prob.p)
const E = length(ode_data[1])


for i = 1:100

    # Generate the noisy data
    noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data]

    function vec_to_params(x)
        e0 = x[1]
        i0 = x[2]
        u0 = [1.0 - e0 - i0, e0, i0, zero(i0)]
        p = x[3:5]
        return (u0 = u0, p = p, σ² = exp(x[6]), κ² = exp(x[7]))
    end
    function loss(x, other_params)
        @unpack u0, p, σ², κ² = vec_to_params(x)
        @unpack prob, ode_data, tsteps, proj, tstops = other_params
        data = (t = tsteps, u = noisy_ode_data)
        return exact_nll(
            remake(prob, u0 = u0, p = p),
            data,
            σ²,
            κ²,
            ;
            tstops = tstops,
            proj = proj,
        )
    end
    κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps, proj)
    e0 = abs(1e-2 * randn())
    i0 = abs(1e-2 * randn())
    p0 = [rand(), rand(), rand()]
    x0 = [e0, i0, p0..., log(1.0), log(κ²0)]

    opt_params = (
        prob = prob,
        ode_data = noisy_ode_data,
        tsteps = tsteps,
        # dt=0.2,
        proj = proj,
        tstops = 0:0.2:100,
    )
    loss(x0, opt_params)

    # loss2(x, params) = loss([x0[1:end-2]..., x...], params)
    # loss2(x0[end-1:end], opt_params)
    # cb2() = (j=0; (x, args...) -> (j+=1; @info "[$i] Iteration $j noise/diff opt" exp.(x); false))
    # f = OptimizationFunction(loss2, GalacticOptim.AutoForwardDiff())
    # optprob = OptimizationProblem(
    #     f, x0[end-1:end], opt_params;
    #     lb=[log(1e-6), log(1e-20)],
    #     ub=[log(1e2), log(1e20)],
    # )
    # optsol = solve(optprob, LBFGS(); maxiters=1000, cb=cb2)
    # x0 = [x0[1:end-2]..., optsol.u...]


    f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    optprob = OptimizationProblem(
        f,
        x0,
        opt_params;
        lb = [0.0, 0.0, 0.0, 0.0, 0.0, log(1e-6), log(1e-20)],
        ub = [1.0, 1.0, 1.0, 1.0, 1.0, log(1e2), log(1e20)],
    )

    @info "[$i] Optimize"
    _cb() = (
        j = 0;
        (x, l, args...) -> begin
            j += 1
            @unpack u0, p, σ², κ² = vec_to_params(x)
            @info "[$i] Iteration $j" loss = l u0 prob.u0 p prob.p σ² κ²
            false
        end
    )
    optsol = solve(
        optprob,
        LBFGS(linesearch = Optim.LineSearches.BackTracking());
        maxiters = 1000,
        cb = _cb(),
    )

    @unpack u0, p, σ², κ² = vec_to_params(optsol.u)
    @info "[$i] Done" u0_inf = u0 u0_true = prob.u0 p_inf = p p_true = prob.p
    open(filepath, "a") do f
        write(f, join(string.([u0..., p...]), ",") * "\n")
    end
end
