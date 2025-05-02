using Revise
using CPOMDPExperiments
using Infiltrator
using ProgressMeter
using Distributed
using Random

nsims = 100
run = [false, false, true] #(pomcpow, pomcp, pft-dpw)

cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew();λ=[1.])

# global parameters
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

if run[1] # POMCPOW
    kwargs = Dict(
        :tree_queries=>tree_queries, 
        :k_observation => k_observation, # 0.1,
        :alpha_observation => alpha_observation, #0.5,
        :enable_action_pw => false,
        :check_repeat_obs => false,
        :max_depth => max_depth,
        :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
        :estimate_value=>zeroV_trueC,
    )
    # exp1 = LightExperimentResults(nsims)
    exp = []
    reached = []
    avoided = []
    for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, solver.rng), 
            planner)
        exp1 = run_cpomdp_simulation(cpomdp, solver, updater;track_history=true)
        push!(reached, abs(exp1[1][end][:s].y) <= 1.0)
        push!(avoided, sum([(exp1[1][k][:s].y > 10.) for k=1:length(exp1[1])]) < 1)
        println("R ", reached)
        println("A ", avoided)
    end
end

if run[3] # PFT
    kwargs = Dict(
        :n_iterations=>pft_tree_queries, 
        :k_state => k_observation, # 0.1,
        :alpha_state => alpha_observation, #0.5,
        :enable_action_pw => false,
        :check_repeat_state => false,
        :depth => max_depth,
        :exploration_constant => c,
        :nu => nu, 
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(asched),
        :estimate_value=>CPOMDPExperiments.heuristicV,
    )
    reached2 = []
    avoided2 = []
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        rng = MersenneTwister(i)
        up = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, pf_filter_size, rng)
        solver = CPOMDPExperiments.CMCTS.BeliefCMCTSSolver(
            CPOMDPExperiments.CMCTS.CDPWSolver(;kwargs..., rng=rng),
            up)
        updater(planner) = CPOMDPExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, rng), 
            planner) 
        exp3 = run_cpomdp_simulation(cpomdp, solver, updater;track_history=true)
        push!(reached2, abs(exp3[1][end][:s].y) <= 1.0)
        push!(avoided2, sum([(exp3[1][k][:s].y > 10.) for k=1:length(exp3[1])]) < 1)
        println("R ", reached2)
        println("A ", avoided2)
    end
    print_and_save(exp3,"results/lightdark_pft_$(nsims)sims.jld2")
end