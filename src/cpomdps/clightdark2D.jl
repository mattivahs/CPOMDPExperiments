# A one-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg

import Base: ==, +, *, -
import POMDPs: discount, isterminal, actions, initialstate, initialobs, observation, transition, reward, convert_s

struct LightDark2DState
    status::Int64
    x::Float64
    y::Float64
end



mutable struct LightDark2D <: POMDPs.POMDP{LightDark2DState, Int, Float64}
    discount_factor::Float64
    correct_r::Float64
    incorrect_r::Float64
    step_size::Float64
    movement_cost::Float64
    sigma::Any
    actions::Vector{Vector{Float64}}
end

default_actions = [[-1., -1.], [-1., 0.], [-1., 1.], [0., -1.], [0., 0.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]]

default_sigma(s::LightDark2DState) = sqrt((s.x - 0)^2 + (s.y - 0.8)^2) + 1e-4

LightDark2D() = LightDark2D(0.99, 100.0, -10.0, 0.1, 0.0, default_sigma, default_actions)

discount(p::LightDark2D) = p.discount_factor

isterminal(::LightDark2D, act::Int64) = act == 0

isterminal(::LightDark2D, s::LightDark2DState) = s.status < 0


actions(::LightDark2D) = 1:9


struct LDUniformStateDist
    min::Float64
    max::Float64
end


dd = LDUniformStateDist(-1, 1)

sampletype(::Type{LDUniformStateDist}) = LightDark2DState
# rand(rng::AbstractRNG, d::LDUniformStateDist) = LightDark2DState(0, d.min + rand(rng)*(d.max - d.min), d.min + rand(rng)*(d.max - d.min))

import Base: rand
rand(rng::AbstractRNG, d::LDUniformStateDist) = LightDark2DState(0, d.min + rand(rng)*(d.max - d.min), d.min + rand(rng)*(d.max - d.min))


import POMDPs: initialstate
initialstate(pomdp::LightDark2D) = LDUniformStateDist(-1, 1)
# initialstate(pomdp::LightDark2D, rng) = LightDark2DState(0, dd.min + rand(rng)*(dd.max - dd.min), d.min + rand(rng)*(dd.max - dd.min))
initialobs(m::LightDark2D, s) = observation(m, s)

observation(p::LightDark2D, sp::LightDark2DState) = Normal(sqrt((sp.x - 0)^2 + (sp.y - 0.8)^2), p.sigma(sp))

function transition(p::LightDark2D, s::LightDark2DState, a::Int)
    act = p.actions[a]
    if a == [0., 0.]
        return Deterministic(LightDark2DState(-1, s.x, s.y))
    else
        return Deterministic(LightDark2DState(s.status, s.x+act[1]*p.step_size, s.y+act[2]*p.step_size))
    end
end

function reward(p::LightDark2D, s::LightDark2DState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == [0., 0.]
        if sqrt((s.x - 1.8)^2 + (s.y)^2) < 0.2
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end


convert_s(::Type{A}, s::LightDark2DState, p::LightDark2D) where A<:AbstractArray = eltype(A)[s.status, s.x, s.y]
convert_s(::Type{LightDark2DState}, s::A, p::LightDark2D) where A<:AbstractArray = LightDark2DState(Int64(s[1]), s[2], s[3])


## CPOMDP

struct CLightDark2D{P<:LightDark2D,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    cost_budget::Float64
    max_x::Float64
    min_y::Float64
    max_y::Float64
end

function CLightDark2D(;pomdp::P=LightDark2D(),
    cost_budget::Float64=0.,
    max_x::Float64=1.,
    min_y::Float64=-0.3,
    max_y::Float64=0.3,
    ) where {P<:LightDark2D}
    return CLightDark2D{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget,max_x, min_y,max_y)
end

costs(p::CLightDark2D, s::LightDark2DState, a::Int) = Float64[(s.x >= p.max_x && s.y>=p.max_y) || (s.x >= p.max_x && s.y<=p.min_y)]
costs_limit(p::CLightDark2D) = [p.cost_budget]
n_costs(::CLightDark2D) = 1
max_reward(p::CLightDark2D) = p.pomdp.correct_r
min_reward(p::CLightDark2D) = -p.pomdp.movement_cost


function heuristicV(p::POMDP, s::ParticleFilters.ParticleCollection{S}) where {S<:LightDark2DState}
    ps = [[p.x, p.y] for p in particles(s)]
    m = Statistics.mean(ps)
    return norm(m - [1.8, 0.0])

end

heuristicV(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, 
    args...) where {P<:CLightDark2D, S<:LightDark2DState} = return (heuristicV(p.cpomdp.pomdp, s))

heuristicV(p::POMDPTools.GenerativeBeliefMDP{P}, s::ParticleFilters.ParticleCollection{S}, 
    args...) where {P<:CLightDark2D, S<:LightDark2DState} = return heuristicV(p.pomdp, s)