using CollisionAvoidancePOMDPs

@with_kw mutable struct CollisionAvoidancePOMDP <: POMDP{Vector{Float64}, Float64, Vector{Float64}}
    h_rel_range::Vector{Real} = [-150, 150] # initial relative altitudes [m]
    dh_rel_range::Vector{Real} = [-1e-6, 1e-6] # initial relative vertical rates [m²]
    ddh_max::Real = 1.0                     # vertical acceleration limit [m/s²]
    τ_max::Real = 40                        # max time to closest approach [s]
    actions::Vector{Real} = [0.0, -5, 5]    # relative vertical rate actions [m/s²]
    a_prev_zero::Bool = true                # whether to update `a_prev` when the action is zero
    collision_threshold::Real = 50          # collision threshold [m]
    reward_collision::Real = -100           # reward obtained if collision occurs
    reward_reversal::Real = -1              # reward obtained if action reverses direction (e.g., from +5 to -5)
    reward_alert::Real = -1                 # reward obtained if alerted (i.e., non-zero vertical rates)
    apply_continuous_alerting_cost::Bool = false # apply penalty during any alert, not just the first alert
    apply_min_separation_cost::Bool = false # apply penalty based on separation to be minimized
    px = DiscreteNonParametric([1, 0.0, -1], [0.25, 0.5, 0.25]) # transition noise on relative vertical rate [m/s²]
    σobs::Vector{Real} = [15, 1, eps(), eps()] # observation noise [h_rel, dh_rel, a_prev, τ]
    γ::Real = 0.99                          # discount factor
end


function POMDPs.reward(pomdp::CollisionAvoidancePOMDP, s, a)
    h_rel, dh_rel, a_prev, τ = s

    r = 0.0
    is_nmac, is_alert, is_reversal = isfailure(pomdp, s, a)
    if pomdp.apply_continuous_alerting_cost
        if a != 0
            # Alerting
            r += pomdp.reward_alert
        end
    else
        if is_alert
            # Alerted
            r += pomdp.reward_alert
        end
    end
    if is_reversal
        # Reversed the action
        r += pomdp.reward_reversal
    end
    if pomdp.apply_min_separation_cost && abs(τ) < eps()
        r += -abs(h_rel) # minimize separation
    end
    return r
end

struct CCAS{P<:CollisionAvoidancePOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    cost_budget::Float64
    collision_threshold::Real
end

function CCAS(;pomdp::P=CollisionAvoidancePOMDP(),
    cost_budget::Float64=0.5,
    collision_threshold::Real=50
    ) where {P<:CollisionAvoidancePOMDP}
    return CCAS{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget,collision_threshold)
end

costs(p::CCAS, s::Vector{Float64}, a::Float64) = Float64[isfailure(p, s, a)[1]]
costs_limit(p::CCAS) = [p.cost_budget]
n_costs(::CCAS) = 1

function isfailure(pomdp::CCAS, s, a, sp=missing)
    h_rel, dh_rel, a_prev, τ = s
    is_nmac = abs(h_rel) < pomdp.collision_threshold && τ < eps()
    is_alert = a_prev == 0 && a != 0 # alert
    is_reversal = a_prev != 0 && a != 0 && a != a_prev # reversal
    return [is_nmac, is_alert, is_reversal]
end

max_reward(p::CCAS) = 0.
min_reward(p::CCAS) = -100.

