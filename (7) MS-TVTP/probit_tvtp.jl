# Gibbs sampler: 2-state Bayesian Markov-switching autoregression w/ time-varying transition probability

using Random, Distributions, StatsBase, StatsFuns
using LinearAlgebra
using Plots, StatsPlots, LaTeXStrings
using ProgressMeter

symmetric(_m) = Symmetric((_m + _m') ./ 2)
default(size=(500,200))
Random.seed!(1)

"""
Simulate data
"""

T = 300
True = Dict(
    :β => Dict(1 => [-1, 0.7], 2 => [1, -0.7]), # measurement eq. coef.
    :σ²_y => Dict(1 => 0.3, 2 => 1.), # measurement variance
    
    :z => rand(Uniform(-5, 5), T, 2), # transition eq. predictors
    :β_z => [-1, 0.7, -0.7], # transition eq. coef.
    :α_z => 1, # transition eq. intercept
    
    :s => [1; zeros(T-1)], # discrete latent states
    :y => zeros(T), # observation
    :X => zeros(T), # measurement eq. predictors
    
    :p_11 => zeros(T), # P(s_{t} = 1 | s_{t-1} = 1)
    :p_21 => zeros(T), # P(s_{t} = 2 | s_{t-1} = 1)
    :p_12 => zeros(T), # P(s_{t} = 1 | s_{t-1} = 2)
    :p_22 => zeros(T), # P(s_{t} = 2 | s_{t-1} = 2)
)

# Probit probabilities
f(x, y) = [1, x, y]' * True[:β_z] # given s_{t-1} = 1
g(x, y) = [1, x, y]' * True[:β_z] + True[:α_z] # given s_{t-1} = 1

# Simulate data
for t in 2:T
    True[:p_21][t] = cdf(Normal(), f(True[:z][t,:]...))
    True[:p_11][t] = 1 - True[:p_21][t]
    
    True[:p_22][t] = cdf(Normal(), g(True[:z][t,:]...))
    True[:p_12][t] = 1 - True[:p_22][t]
    
    if True[:s][t-1] == 1
        True[:s][t] = DiscreteNonParametric(
            [1, 2], [True[:p_11][t], True[:p_21][t]]) |> rand
    elseif True[:s][t-1] == 2
        True[:s][t] = DiscreteNonParametric(
            [1, 2], [True[:p_12][t], True[:p_22][t]]) |> rand
    end
    
    True[:X][t] = True[:y][t-1]
    True[:y][t] = [1; True[:X][t]]' * True[:β][True[:s][t]]
    True[:y][t] += rand(Normal(0, True[:σ²_y][True[:s][t]] |> sqrt))
end

# Visualize data
plot(
    plot(True[:y], color=:black),
    plot(True[:s], label="", yticks=[1;2]),
    layout=grid(2,1), label="",
    ylabel=[L"y_t" L"s_t"],
    xlabel=["" "Time"],
)


"""
MCMC algorithm
"""

data = Dict(
    :y => True[:y],
    :X => [ones(T) True[:X]],
    :Z => [ones(T) True[:z]],
)

function s_hamiltonFilter(c) # Sampler: discrete latent states
    P(; t) = [
        c[:p_11][t] c[:p_12][t]
        c[:p_21][t] c[:p_22][t]
    ]
    
    ξ = zeros(T, 2)
    A = [
        I(2) - P(t=1);
        ones(2)'
    ]
    ξ_11 = (A' * A) \ (A' * [0, 0, 1])
    
    """Forward-filtering"""
    ℓ = 0
    for t in 1:T
        ξ_10 = P(t=t) * ξ_11
        f_1 = pdf(Normal(data[:X][t,:]' * c[:β_1], c[:σ²_y][1] |> sqrt), data[:y][t])
        f_2 = pdf(Normal(data[:X][t,:]' * c[:β_2], c[:σ²_y][2] |> sqrt), data[:y][t])
        ξ_11 = ξ_10 .* [f_1, f_2]
        _ℓ = sum(ξ_11)
        ξ_11 = ξ_11 ./ _ℓ
        ξ[t,:] = ξ_11
        ℓ += log(_ℓ)
    end
    
    """Backward-sampling"""
    s_sample = Vector{Int}(undef, T)
    s_sample[T] = DiscreteNonParametric([1,2], ξ[T,:]) |> rand
    for t in T-1:-1:1
        _P = P(t=t+1)[s_sample[t+1],:]
        _ξ = _P .* ξ[t,:]
        s_sample[t] = DiscreteNonParametric([1,2], _ξ / sum(_ξ)) |> rand
    end
    
    c[:s] = s_sample
    c
end

function s0_gibbs(c)
    P_1 = [
        c[:p_11][1] c[:p_12][1]
        c[:p_21][1] c[:p_22][1]
    ]
    A = [
        I(2) - P_1;
        ones(2)'
    ]
    s_0_sample = Categorical((A' * A) \ (A' * [0, 0, 1])) |> rand
    c[:s_0] = s_0_sample
    c
end

function s_star_gibbs(c) # Sampler: latent Gaussian augmentation (Albert and Chib, 1993)
    
    s_lag = [c[:s_0]; c[:s][1:T-1]] .== 2
    _μ_s_star = [data[:Z] s_lag] * [c[:β_z]; c[:α_z]]
    
    s_star_sample = zeros(T)
    for t in 1:T
        if c[:s][t] == 2
            s_star_sample[t] = Truncated(Normal(_μ_s_star[t], 1), 0, Inf) |> rand
        elseif c[:s][t] == 1
            s_star_sample[t] = Truncated(Normal(_μ_s_star[t], 1), -Inf, 0) |> rand
        end
    end
    c[:s_star] = s_star_sample
    c
end

function P_calc(c) # Compute transition matrix
    _μ_s_star = data[:Z] * c[:β_z]
    c[:p_11] = cdf(Normal(), -_μ_s_star)
    c[:p_21] = 1 .- c[:p_11]
    c[:p_12] = cdf(Normal(), -(_μ_s_star .+ c[:α_z]))
    c[:p_22] = 1 .- c[:p_12]
    c
end

function βz_gibbs(c) # Sampler: transition eq. coef.s
    
    s_lag = [c[:s_0]; c[:s][1:T-1]] .== 2 # (2, 1) => (1, 0)
    _Z = [data[:Z] s_lag]
    
    β_z_sample = MvNormalCanon(
        priors[:β_z].h + (_Z' * c[:s_star]),
        (priors[:β_z].J + (_Z' * _Z)) |> symmetric
    ) |> rand
    
    c[:β_z], c[:α_z] = β_z_sample[1:end-1], β_z_sample[end]
    c
end

function β_gibbs(c) # Sampler: measurement eq. coef.s
    for i in 1:2
        _idx = findall(c[:s] .== i)
        _y = data[:y][_idx]
        _X = data[:X][_idx,:]
        
        J = priors[:β].J + (_X' * _X) / c[:σ²_y][i]
        h = priors[:β].J * mean(priors[:β]) + (_X' * _y) / c[:σ²_y][i]
        
        β_sample = zeros(2)
        if i == 1
            β_sample[:] = MvNormalCanon(h, J |> symmetric) |> rand
            c[:β_1] = β_sample
        elseif i == 2
            V = inv(J)
            m = V * h
            
            β_sample[2:end] = MvNormal(m[2:end], V[2:end,2:end] |> symmetric) |> rand
            m_cond = m[1:1] + V[1:1,2:end] * (V[2:end,2:end] \ (β_sample[2:end] - m[2:end]))
            V_cond = V[1:1,1:1] - V[1:1,2:end] * (V[2:end,2:end] \ V[2:end,1:1])
            β_sample[1] = Truncated(Normal(
                m_cond[1], V_cond[1,1] |> sqrt),
                c[:β_1][1], Inf
            ) |> rand
            c[:β_2] = β_sample
        end
    end
    
    c
end

function σy_gibbs(c) # Sampler: transition eq. variances
    for i in 1:2
        _idx = findall(c[:s] .== i)
        _y = data[:y][_idx]
        _X = data[:X][_idx,:]
        e = _y - _X * c["β_$(i)" |> Symbol]
        c[:σ²_y][i] = InverseGamma(
            shape(priors[:σ²_y]) + length(_idx) / 2,
            scale(priors[:σ²_y]) + (e' * e) / 2
        ) |> rand
    end
    c
end

# Set priors
priors = Dict(
    :β => MvNormalCanon(I(2) / 10),
    :β_z => MvNormalCanon(I(4) / 1),
    :σ²_y => InverseGamma(11, 1),
)

# Initial values for MCMC
params = Dict(
    :β_1 => True[:β][1],
    :β_2 => True[:β][2],
    :β_z => zeros(3), :α_z => 0.,
    
    :σ²_y => [1., 1.],
    
    :p_11 => repeat([0.5], T),
    :p_21 => repeat([0.5], T),
    :p_22 => repeat([0.5], T),
    :p_12 => repeat([0.5], T),
    
    :s => ones(T),
    :s_0 => 1,
    :s_star => zeros(T),
)

"""
Run MCMC
"""
gibbs = σy_gibbs ∘ β_gibbs ∘ βz_gibbs ∘ P_calc ∘ s_star_gibbs ∘ s0_gibbs ∘ s_hamiltonFilter
R = 2000 # MCMC iterations
Θ = []
@showprogress for r in 1:R
    params |> gibbs
    if (r > 1000) & (r % 10 == 0)
        push!(Θ, copy(params))
    end
end

"""
Posterior visualization
"""
chain = Dict(
    _key => cat([Θ[r][_key] for r in 1:length(Θ)
                ]...; dims=4)
    for _key in keys(params)
)

# Measurement eq. coef.s
begin
	s = 1
	@assert 1 <= s <= 2
	plot(chain["β_$(s)" |> Symbol][:,1,1,:]', label="")
	hline!(True[:β][s], label="", color=:black, style=:dash)
end

# Transition eq. coef.s
begin
	plot([chain[:β_z][:,1,1,:]' chain[:α_z][1,1,1,:]], label="")
	hline!([True[:β_z]; True[:α_z]], label="", color=:black, style=:dash)
end

# Discrete latent states
plot(
	heatmap(True[:s][:,:]),
	heatmap(chain[:s][:,1,1,:]),
	title=["True" "MCMC"]
)