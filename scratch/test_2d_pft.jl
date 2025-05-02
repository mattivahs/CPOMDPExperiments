using Revise
using CPOMDPExperiments
using D3Trees
using Plots 

nu = 0.0
asched = 0.5

### Find Good Settings
kwargs = Dict(:n_iterations=>Int(1e2), 
        :k_state => 5., 
        :alpha_state => 1/15, 
        :enable_action_pw=>false,
        :depth => 10,
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(0.5),
        :exploration_constant => 90., #90.
        :nu => nu, 
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(asched),
        ) 

nu = 0.0
λ_test = [1.]

npart = Int(10)
cpomdp = SoftConstraintPOMDPWrapper(CLightDark2D();λ=λ_test)

up = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, npart)

solver = CPOMDPExperiments.CMCTS.BeliefCMCTSSolver(
    CPOMDPExperiments.CMCTS.CDPWSolver(;kwargs..., nu=nu), up)

updater(planner) = CPOMDPExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
    CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.solver.rng), 
    planner)

hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

R3
C3[1]
RC3
