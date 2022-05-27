function lotkavolterra(;
    u0=[5.0, 3.0],
    tspan=(0.0, 2.0),
    p=[2.0, 1.0, 4.0, 1.0],
    tsteps=0:0.1:2,
)
    d_p = length(p)
    d_u = length(u0)
    function f(du, u, p, t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = α * x - β * x * y
        du[2] = dy = -δ * y + γ * x * y
        return nothing
    end
    prob = ODEProblem(f, u0, tspan, p)
    prob = ProbNumDiffEq.remake_prob_with_jac(prob)
    sol = solve(prob, Vern9(), abstol=1e-9, reltol=1e-6, saveat=tsteps)

    get_random_parameter() = abs.(randn(d_p))
    parameter_bounds = (zeros(d_p), 100*ones(d_p))
    u0_bounds = (zeros(d_u), 100*ones(d_u))

    noise_levels = Dict(
        "low" => 0.1^2,
        "high" => 0.5^2,
    )

    return prob, (sol.t, sol.u), noise_levels, get_random_parameter, parameter_bounds, u0_bounds
end


function fitzhughnagumo(;
                       u0=[-1.0, 1.0],
                       tspan=(0.0, 10.0),
                       p=[0.2, 0.2, 3.0],
                       tsteps=range(tspan[1], tspan[2], length=20),
                       )
    d_p = length(p)
    d_u = length(u0)
    function fhn(du, u, p, t)
        V, R = u
        du[1] = p[3] * (u[1] - u[1]^3/3 + u[2])
        du[2] = -1/p[3] * (u[1] - p[1] + p[2]*u[2])
        return nothing
    end
    prob = ODEProblem(fhn, u0, tspan, p)
    prob = ProbNumDiffEq.remake_prob_with_jac(prob)
    sol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10, saveat=tsteps)

    get_random_parameter() = abs.(randn(d_p))
    parameter_bounds = (zeros(d_p), 100*ones(d_p))
    u0_bounds = (-100*ones(d_u), 100*ones(d_u))

    snr2noise(SNR) = begin
        std_devs_signal = [std([u[i] for u in sol.u]) for i in 1:d_u]
        std_devs_noise = std_devs_signal / sqrt(SNR)
        noise_var = std_devs_noise .^ 2
        return noise_var
    end

    noise_levels = Dict(
        # "low" => snr2noise(100),
        # "high" => snr2noise(10),
        "low" => 0.005,
        "high" => 0.05,
    )

    return prob, (sol.t, sol.u), noise_levels, get_random_parameter, parameter_bounds, u0_bounds
end

function protein_transduction(;
                              u0=[1.0, 0.0, 1.0, 0.0, 0.0],
                              tspan=(0.0, 100.0),
                              p=[0.07, 0.6, 0.05, 0.3, 0.017, 0.3],
                              tsteps=[0, 1, 2, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100],
                              )
    d_p = length(p)
    d_u = length(u0)
    function pt(du, u, p, t)
        S, dS, R, R_S, R_pp = u
        du[1] = dS = -p[1]*S - p[2]*S*R + p[3]*R_S
        du[2] = ddS = p[1]*S
        du[3] = dR = -p[2]*S*R + p[3]*R_S + p[5] * R_pp / (p[6] + R_pp)
        du[4] = dR_S = p[2]*S*R - p[3]*R_S - p[4]*R_S
        du[5] = dR_pp = p[4]*R_S - p[5] * R_pp / (p[6] + R_pp)
        return nothing
    end
    prob = ODEProblem(pt, u0, tspan, p)
    prob = ProbNumDiffEq.remake_prob_with_jac(prob)
    sol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10, saveat=tsteps)

    get_random_parameter() = abs.(randn(d_p))
    parameter_bounds = (1e-8*ones(d_p), 10*ones(d_p))
    u0_bounds = (zeros(d_u), ones(d_u))

    noise_levels = Dict(
        "low" => 0.001^2,
        "high" => 0.01^2,
    )

    return prob, (sol.t, sol.u), noise_levels, get_random_parameter, parameter_bounds, u0_bounds
end
