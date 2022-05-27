function DataCondCB(times, vals, ode_data_noise, saveinto=nothing)
    function affect!(integ)
        val = vals[integ.t.==times, :][:]

        E0 = integ.cache.Proj(0)
        # @assert integ.u ≈ E0 * integ.cache.x.μ

        x = integ.cache.x
        pu = E0 * x

        # Compute the negative log-likelihood, before conditioning
        nll = -logpdf(Gaussian(pu.μ, Matrix(pu.Σ) + ode_data_noise * I), val)
        if !isnothing(saveinto)
            saveinto[1] += nll
        end

        # Condition on the data
        m, P = x.μ, x.Σ
        z = E0 * m
        H = E0
        S = H * P * H' + ode_data_noise * I
        S_inv = inv(S)
        K = P * H' * S_inv
        mnew = m + K * (val .- z)
        Pnew = X_A_Xt(P, (I - K * H))
        # Overwrite the current state and solution
        copy!(m, mnew)
        copy!(P, Pnew)
        integ.u = E0 * x.μ
        return nothing
    end
    return PresetTimeCallback(times, affect!)
end

function approximate_nll(prob, ode_data, tsteps, ode_data_noise)
    function loss(θ)
        θ_pos = exp.(θ)
        p, σ² = θ_pos[1:4], θ_pos[5]

        nll = [zero(eltype(θ))]
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedDiffusion(σ², false), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
            p=p,
            # maxiters=Int(1e4),
            callback=DataCondCB(tsteps, ode_data, ode_data_noise, nll),
        )
        if sol.retcode != :Success
            return 999_999_999, sol
        end
        return nll[1], sol
    end
    return loss
end
