using ProbNumDiffEq, OrdinaryDiffEq, GalacticOptim, Optim, Plots, LinearAlgebra, Statistics
using Fenrir
using UnPack


############################################################################################
# Setup
############################################################################################
t0, tmax, dt = 0.0, 10.0, 0.01
proj = [0 1]

const g = 9.81
u₀ = [0,π/2]
tspan = (0.0,6.3)
function simplependulum(du,u,p,t)
    L = p
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*sin(θ)
end
# prob, (tsteps, ode_data), noise_levels, θ_init, θ_bounds, u0_bounds = lotkavolterra(
tsteps = t0:dt:tmax
tspan = (t0, tmax)
prob = ODEProblem(simplependulum, u₀, tspan, 1.0)
sol = solve(prob,Tsit5())
ode_data = [proj * u for u in sol.(tsteps)]
noise_var = 0.1
noisy_ode_data = [u + sqrt.(noise_var) .* randn(size(u)) for u in ode_data];
plot(sol)
scatter!(tsteps, ProbNumDiffEq.stack(noisy_ode_data))



function loss_classic(p)
    sol = solve(prob, Tsit5(), p=p, abstol=1e-6, reltol=1e-6, saveat=tsteps)
    return mean(norm.([proj*u for u in sol.u] - noisy_ode_data))
end
ps = 0.01:0.01:5.0
plot(loss, ps)



function loss_fenrir(p, κ²_log10)
    @info "p=$p log_10(κ²)=$(κ²_log10)"
    dt = 1e-2
    data = (t=tsteps, u=noisy_ode_data)
    l, _, _ = exact_nll(remake(prob, p=p), data, noise_var, 10.0 ^ κ²_log10; dt=dt, proj=proj)
    return l
end


kappas_log10 = 10:0.1:30
ps = 0.05:0.05:10.0
losses = zeros(length(ps), length(kappas_log10))
indices = [(i,j) for i in 1:length(ps), j in 1:length(kappas_log10)]
for (i, j) in indices
    losses[i,j] = loss_fenrir(ps[i], kappas_log10[j])
end
heatmap(losses)


# Can I evaluate the gradients, too?
using ForwardDiff
_l(x) = loss_fenrir(x...)
gradients = zeros(length(ps), length(kappas_log10), 2)
indices = [(i,j) for i in 1:length(ps), j in 1:length(kappas_log10)]
for (i, j) in indices
    gradients[i,j,:] = ForwardDiff.gradient(_l, [ps[i], kappas_log10[j]])
end


using JLD
save("experiments/4/loss_heatmap.jld", "losses", losses, "ps", ps, "kappas_log10", kappas_log10, "gradients", gradients)
