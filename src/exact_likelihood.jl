"""
Compute the exact negative log-likelihood (NLL) of data given a problem and algorithm
"""
function exact_nll(
    ode_problem::ODEProblem,
    data,
    observation_noise_var,
    diffusion_var,
    ;
    dt=false,
    proj=I,
    order=5,
    tstops=[],
    )
    N = length(data.u)
    D = length(ode_problem.u0)
    E = length(data.u[1])
    P = length(ode_problem.p)

    σ² = observation_noise_var

    κ² = diffusion_var
    alg = EK1(
        order=order,
        diffusionmodel=FixedMVDiffusion(κ², false),
        smooth=false,
    )::EK1{0, true, Val{:forward}, FixedMVDiffusion{typeof(κ²)}, TaylorModeInit}


    ########################################################################################
    # Part 1: Forward-solve the ODE
    ########################################################################################
    integ = init(
        ode_problem,
        alg,
        dense=false,
        tstops=union(data.t, tstops),
        adaptive=false, dt=dt,
        # abstol=1e-9, reltol=1e-9,
        # save_everystep=true, saveat=data.t,
    )
    sol = solve!(integ)

    if sol.retcode != :Success
        # What should be done if the solver does not succeed? Raise error? Return large NLL?
        @error "The PN ODE solver did not succeed!"
        # error()
        return 1e10, sol.t, sol.pu
    end

    ########################################################################################
    # Part 2: Backwards-iterate: Compute the NLL and condition on the data
    ########################################################################################
    NLL, times, states = backwards_iterate!(integ, sol, observation_noise_var, data; proj=proj)
    # Finally, project the state estimates to the zeroth derivative for plotting purposes
    u_probs = project_to_solution_space!(integ.sol.pu, states, integ.cache.SolProj)
    return NLL, times, u_probs

end
function backwards_iterate!(integ, sol,
                           observation_noise_var,
                           data; proj=I)
    N = length(data.u)
    D = length(integ.u)
    E = length(data.u[1])
    P = length(integ.p)

    R = Diagonal(observation_noise_var .* ones(E))

    NLL = zero(eltype(integ.p))

    @unpack A, Q, x_tmp, x_tmp2, m_tmp = integ.cache
    x_tmp3 = integ.cache.x
    m_tmp = get_lowerdim_measurement_cache(m_tmp, E)

    x_pred = sol.x_pred # This contains the predicted states of the forward pass
    x_smooth = sol.x_filt # These will be smoothed in the following
    diffusion = sol.diffusions[1] # diffusions are all the same anyways

    H = proj * integ.cache.SolProj
    ZERO_DATA = zeros(E)

    # First update on the last data point
    if sol.t[end] in data.t
        NLL += compute_nll_and_update!(x_smooth[end], data.u[end], H, R, m_tmp, ZERO_DATA)
    end

    # Now iterate backwards
    for i in length(x_smooth)-1:-1:1
        dt = sol.t[i+1] - sol.t[i]
        ProbNumDiffEq.make_preconditioners!(integ.cache, dt)
        P, PI = integ.cache.P, integ.cache.PI

        xf = mul!(x_tmp, P, x_smooth[i])
        xs = mul!(x_tmp2, P, x_smooth[i+1])
        xp = mul!(x_tmp3, P, x_pred[i+1])
        ProbNumDiffEq.smooth!(xf, xs, xp, A, Q, integ, diffusion)
        xs = mul!(x_smooth[i], PI, xf)

        if sol.t[i] in data.t
            data_idx = findall(x -> x == sol.t[i], data.t)[1]
            NLL += compute_nll_and_update!(xs, data.u[data_idx], H, R, m_tmp, ZERO_DATA)
        end
    end

    return NLL, sol.t, x_smooth
end

function get_lowerdim_measurement_cache(m_tmp, E)
    _z, _S = m_tmp
    return Gaussian(
        view(_z, 1:E),
        SRMatrix(view(_S.squareroot, 1:E, :), view(_S.mat, 1:E, 1:E))
    )
end

function measure!(x, H, R, m_tmp)
    z, S = m_tmp
    mul!(z, H, x.μ)
    ProbNumDiffEq.X_A_Xt!(S, x.Σ, H)
    _S = S.mat .+= R
    return Gaussian(z, Symmetric(_S))
end
function compute_nll_and_update!(x, u, H, R, m_tmp, ZERO_DATA)
    msmnt = measure!(x, H, R, m_tmp)
    msmnt.μ .-= u
    nll = -logpdf(msmnt, ZERO_DATA)
    copy!(x, ProbNumDiffEq.update(x, msmnt, H))
    return nll
end

function project_to_solution_space!(u_probs, states, projmat)
    for (pu, x) in zip(u_probs, states)
        mul!(pu, projmat, x)
    end
    return u_probs
end


function get_initial_diff(prob, noisy_ode_data, tsteps, proj=I)
    N = length(tsteps)
    E = length(noisy_ode_data[1])

    integ = init(prob, EK1())
    cache = integ.cache

    @unpack P, PI, A, Q, Ah, Qh = integ.cache
    @unpack measurement, x_filt, x_pred = integ.cache
    @unpack K1, K2, x_tmp2, m_tmp = integ.cache

    m_tmp = get_lowerdim_measurement_cache(m_tmp, E)
    measurement = get_lowerdim_measurement_cache(measurement, E)
    K1 = view(K1, :, 1:E)

    H = proj * cache.SolProj
    x = copy(cache.x)
    x.μ .= 0
    D, _ = size(x.Σ)
    σ = 1e2
    x.Σ.squareroot .= σ*I(D)
    x.Σ.mat .= σ^2*I(D)

    t0 = prob.tspan[1]
    asdf = zero(eltype(x.μ))
    for i in 1:N
        dt = i == 1 ? tsteps[i] - t0 : tsteps[i] - tsteps[i-1]

        if dt > 0
            ProbNumDiffEq.make_preconditioners!(cache, dt) # updates P and PI
            @. Ah .= PI.diag .* A .* P.diag'
            ProbNumDiffEq.X_A_Xt!(Qh, Q, PI)

            ProbNumDiffEq.predict!(x_pred, x, Ah, Qh, cache.C1)
        else
            copy!(x_pred, x)
        end

        z, S = measurement
        mul!(z, H, x_pred.μ)
        z .-= noisy_ode_data[i]
        ProbNumDiffEq.X_A_Xt!(S, x_pred.Σ, H)
        ProbNumDiffEq.update!(x, x_pred, measurement, H, K1, x_tmp2.Σ.mat, m_tmp)

        asdf += z' * (S \ z)
    end
    # @assert tsteps[1] == prob.tspan[1]
    return asdf / N
end
