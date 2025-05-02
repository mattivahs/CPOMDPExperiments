using Revise
using CPOMDPExperiments
using Plots
using D3Trees
using Random
using FileIO
using CollisionAvoidancePOMDPs

kwargs = Dict(:tree_queries=>1000)
c = 250.0
nu = 0.0
λ_test = [1.]



# cpomcpow - fixed budget
cpomdp = SoftConstraintPOMDPWrapper(CCAS();λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu),
    )

hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)
println(R3)
println(C3[1])
RC3
hist3
# plot_lightdark_beliefs(hist3,"belief_constrained.png")

