function hodgkinhuxley(;
                       u0=[-70, 0.0, 1.0, 0.1],
                       tspan=(0.0, 30.0),
                       p=[70.0, 10.0, 0.05],
                       tsteps=0:0.2:30,
                       noise=1e-3,
                       )
    function hh_pospischil(du, u, p, t)
        E = u[1]
        m = u[2]
        h = u[3]
        n = u[4]

        V_T = -63.0

        g_Na, g_Kd, g_leak = p

        alpha_act = -0.32 * (E - V_T - 13) / (exp(-(E - V_T - 13) / 4) - 1)
        beta_act = 0.28 * (E - V_T - 40) / (exp((E - V_T - 40) / 5) - 1)
        du[2] = (alpha_act * (1.0 - m)) - (beta_act * m)

        alpha_inact = 0.128 * exp(-(E - V_T - 17) / 18)
        beta_inact = 4 / (1 + exp(-(E - V_T - 40) / 5))
        du[3] = (alpha_inact * (1.0 - h)) - (beta_inact * h)

        alpha_kal = -0.032 * (E - V_T - 15) / (exp(-(E - V_T - 15) / 5) - 1)
        beta_kal = 0.5 * exp(-(E - V_T - 10) / 40)
        du[4] = (alpha_kal * (1.0 - n)) - (beta_kal * n)

        area = 20000.0
        I_Na = -(E - 50) * g_Na * area * (m^3) * h
        I_K = -(E - (-90)) * g_Kd * area * n^4
        I_leak = -(E - (-65)) * g_leak * area

        I_ext = 1.0 * area

        du[1] = (I_leak + I_K + I_Na + I_ext) / (1 * area)
        return nothing
    end

    prob = ODEProblem(hh_pospischil, u0, tspan, p)

    # sol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10, saveat=tsteps)
    # ode_data = Array(sol(tsteps))'
    # ode_data += sqrt(noise) * randn(size(ode_data))

    # return prob, (sol.t, sol.u), noise
    return prob
end


function sird(;
              I0=1e-7,
              u0=[1.0-I0, I0, 0.0, 0.0],
              tspan=(0.0, 100.0),
              p=(β = 0.5, γ = 0.06, η = 0.002),
              )
    function f(du, u, p, t)
        N = 1
        β, γ, η = p
        S, I, R, D = u
        du[1] = dS = - β * S * I / N
        du[2] = dI = β * S * I / N - γ * I - η * I
        du[3] = dR = γ * I
        du[4] = dD = η * I
    end
    prob = ODEProblem(f, u0, tspan, p)
    return prob
end


"""
Partially motivated from https://www.nature.com/articles/s41598-021-97260-0#Sec2
E.g. parameter choices for γ and λ are chosen from there
"""
function seir(;
              I0=1e-5,
              E0=4e-5,
              u0=[1.0-E0-I0, E0, I0, 0.0],
              tspan=(0.0, 100.0),
              p=(
                  β_E = 0.5,
                  # β_I = 0.01,
                  γ = 1/5,
                  λ = 1/21,
              ),
              )
    function f(du, u, p, t)
        N = 1
        β_E, γ, λ = p
        β_I = 0 # Assumption done in the paper, too
        S, E, I, R = u
        du[1] = dS = - (β_E * S * E / N + β_I * S * I / N)
        du[2] = dE = β_E * S * E / N + β_I * S * I / N - γ * E
        du[3] = dI = γ * E - λ * I
        du[4] = dR = λ * I
    end
    prob = ODEProblem(f, u0, tspan, p)
    return prob
end


function pendulum(
    ;
    u0=[0,π/2],
    tspan = (0.0, 10.0),
    p=1.0,
    tsteps=0:0.01:10.0,
    )
    g = 9.81
    function simplependulum(du,u,p,t)
        L = p
        θ = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -(g/L)*sin(θ)
    end
    prob = ODEProblem(simplependulum, u0, tspan, p)
    # prob = ProbNumDiffEq.remake_prob_with_jac(prob)

    sol = solve(prob, Vern9(), abstol=1e-9, reltol=1e-6, saveat=tsteps)

    get_random_parameter() = exp(randn())
    parameter_bounds = (0.0, 100.0)
    u0_bounds = (zeros(2), 10*ones(2))

    noise_var = 0.1

    return prob, (sol.t, sol.u), noise_var, get_random_parameter, parameter_bounds, u0_bounds
end


function logistic(; tspan=(0.0, 3.0), p=[1.0, 1,0], u0=[0.05])
    function f(du, u, p, t)
        du[1] = p[1] * u[1] * (1 - u[1] / p[2])
    end
    prob = ODEProblem(f, u0, tspan, p)
    return prob
end
