import sys
sys.path.append("src")

import numpy as np
import cvxpy as cp
from energyGMM import *
from evChargeState import *
from evMPC import *
from dataAggregator import *

# make base demand model
tr = [19, 32]
gmm = makeModel(timeRange=tr, mixtures=[10])
eGMM = energyGMM(gmm, time_range=tr)

cars = 100
rise = 0.2/cars
dt=0.25
t0 = tr[0]
tf = 31
samps = 30
lam = 1e-2
cr = 1./12
ch0 = 0.4
solver = cp.GUROBI
solver_kwargs = {'TimeLimit': 30.}

naive_costs = []
CE_costs = []
MPC_costs = []

seed_ = 1999
np.random.seed(seed_)
samples = eGMM.sample(samps)
init_state = evChargeState(t=t0, cars=cars, charge=ch0, charge_rate=cr)

# naive_policy
charge_steps = int(1.0/cr*(1-ch0)*1/dt)
action_naive = np.zeros((cars, round((tf-t0)/dt)))
action_naive[:,:charge_steps] = np.ones((cars,charge_steps))

# CE
DCE = np.vstack(
        (eGMM.time_range, eGMM.mle()))
var, cost, constraints = init_state.optimization_problem(
        DCE, tf, rise, lam, dt, relaxation=False)
prob = cp.Problem(cp.Minimize(cost), constraints)
pstar = prob.solve(solver=solver, verbose=True, **solver_kwargs)
action_CE = var.value


for i in range(samps):
    eGMM = energyGMM(gmm, time_range=tr) # reset gmm
    sample = samples[i,:]
    D = np.vstack(
            (eGMM.time_range, sample))

    # Naive
    naive_ecost, naive_tercost, naive_cost = init_state.simulated_cost(D, 
                                                        rise, lam, dt, 
                                                        action_naive)

    # CE
    CE_ecost, CE_tercost, CE_cost = init_state.simulated_cost(D, 
                                                        rise, lam, dt, 
                                                        action_CE)


    # mpc
    mpc = evMPC(eGMM,cars,t=t0, rise=rise, end_times=tf, 
                charge=ch0, charge_rate=cr, eta=lam)
    mpc.simulate(sample, dt)
    action_MPC = mpc.actions_taken
    mpc_ecost, mpc_tercost, mpc_cost = init_state.simulated_cost(D, 
                                                        rise, lam, dt, 
                                                        action_MPC)

    naive_costs.append(naive_ecost)
    CE_costs.append(CE_ecost)
    MPC_costs.append(mpc_ecost)

combined = np.vstack((naive_costs, CE_costs, MPC_costs))
out = combined.T
np.savetxt('%d samples.csv' %(samps),out)