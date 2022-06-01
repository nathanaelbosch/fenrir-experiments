using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using Random

ALG = EK1
DT = 1e-2

# Get the problem and generate the data that we want to fit
prob, (tsteps, ode_data), noise_levels, θ_init, θ_bounds, u0_bounds = lotkavolterra()
D = length(ode_data[1])
noise_var = noise_levels["low"]
rng = MersenneTwister(1337)
# noisy_ode_data = [u + sqrt.(noise_var) .* randn(rng, size(u)) for u in ode_data]

N_RUNS = 100

prob1 = prob

function f2(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y
end
prob2 = ODEProblem(f2, prob.u0, prob.tspan, prob.p)
prob2 = ProbNumDiffEq.remake_prob_with_jac(prob2)

function f3(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x^2 - β * y
    du[2] = dy = -δ * y + γ * x * y
end
prob3 = ODEProblem(f3, prob.u0, prob.tspan, prob.p)
prob3 = ProbNumDiffEq.remake_prob_with_jac(prob3)

function f4(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x^2 - β * y
    du[2] = dy = -δ * y
end
prob4 = ODEProblem(f4, prob.u0, prob.tspan, prob.p)
prob4 = ProbNumDiffEq.remake_prob_with_jac(prob4)

final_NLLs = Vector{Float64}[]

for (i, _prob) in enumerate((prob1, prob2, prob3, prob4))
    push!(final_NLLs, Float64[])
    for j in 1:N_RUNS
        noisy_ode_data = [u + sqrt.(noise_var) .* randn(rng, size(u)) for u in ode_data]

        # Build the named tuple that will be used by the NLL loss function
        opt_params = (
            prob=_prob,
            ode_data=noisy_ode_data,
            tsteps=tsteps,
            noise=noise_var,
            dt=DT,
            Alg=ALG,
        )

        # Initial parameters, with both vector-field parameters and the log-diffusion
        κ²0 = get_initial_diff(prob, noisy_ode_data, tsteps) # * ones(E)
        u0 = clamp.(noisy_ode_data[1], u0_bounds[1], u0_bounds[2])
        p0 = θ_init()
        σ²0 = 1.0
        θ_start = [u0..., p0..., log(σ²0), log(κ²0)]

        @info "[Problem $i][$j]" θ_start
        vec2args(x) = (u0=x[1:D], p=x[D+1:end-2], σ²=exp.(x[end-1]), κ²=exp.(x[end]))
        function loss(x, other_params)
            @unpack u0, p, σ², κ² = vec2args(x)
            @unpack prob, ode_data, tsteps, dt = other_params
            data = (t=tsteps, u=noisy_ode_data)
            return exact_nll(remake(prob, u0=u0, p=p), data, σ², κ²; dt=dt)
        end
        l, times, states = loss(θ_start, opt_params)

        f = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
        optprob = OptimizationProblem(
            f,
            θ_start,
            opt_params;
            lb=[u0_bounds[1]..., θ_bounds[1]..., log(1e-8), log(1e-20)],
            ub=[u0_bounds[2]..., θ_bounds[2]..., log(1e3), log(1e50)],
        )

        @info "[Problem $i][$j] Starting the optimization"
        optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())
        # optimizer = LBFGS()
        optsol = solve(optprob, optimizer; maxiters=200)
        # optsol = solve(optprob, optimizer; maxiters=200, show_trace=true)
        opt_u0, opt_p, opt_σ², opt_κ² =
            optsol.u[1:D], optsol.u[D+1:end-2], exp(optsol.u[end-1]), exp(optsol.u[end])
        @info "[Problem $i][$j] Done" prob.p opt_p opt_u0 opt_σ² opt_κ²
        @info "[Problem $i][$j] Final negative log-likelihood" optsol.minimum

        push!(final_NLLs[end], optsol.minimum)
    end
end

# @info "[Summary] Negative log-likelihoods:" final_NLLs
@assert length(final_NLLs) == 4
@assert length(final_NLLs[1]) == N_RUNS
@assert length(final_NLLs[2]) == N_RUNS
@assert length(final_NLLs[3]) == N_RUNS
@assert length(final_NLLs[4]) == N_RUNS
filepath = "experiments/2/results.txt"
for i in 1:4
    open(filepath, "a") do f
        write(f, "M$i," * join(string.(final_NLLs[i]), ",") * "\n")
    end
end
