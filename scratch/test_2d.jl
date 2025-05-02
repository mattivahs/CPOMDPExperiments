using Pkg
Pkg.activate(".")
using Revise
Pkg.resolve()
using CPOMDPExperiments
using Plots
using D3Trees
using Random
using FileIO
using POMDPs: initialstate


tree_queries = Int(1e4)
pft_tree_queries=Int(1e4)
k_observation = 5.
alpha_observation = 1/15
enable_action_pw = false
max_depth = 10
c = 90.0
nu = 0.0
asched = 0.5
update_filter_size = Int(1e3)
pf_filter_size = 10

kwargs = Dict(
    :tree_queries=>tree_queries, 
    :k_observation => k_observation, # 0.1,
    :alpha_observation => alpha_observation, #0.5,
    :enable_action_pw => false,
    :check_repeat_obs => false,
    :max_depth => max_depth,
    :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
)
λ_test = [1.]



# cpomcpow - fixed budget
cpomdp = SoftConstraintPOMDPWrapper(CLightDark2D();λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu),
    )
# updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
#     CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
#     planner)
hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver)#, updater)
println(R3)
println(C3[1])
RC3
hist3
# plot_lightdark_beliefs(hist3,"belief_constrained.png")

