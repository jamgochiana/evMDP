import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from dataAggregator import *
from energyGMM import *

DEFAULT_PARAMS = {
    'cars': 20,                 # number of cars
    'timeRange': [14, 34],      # time range to run simulation
    'actRate': 1./4,            # step time in hours
    'chargeRate': 1./12,        # level 1 full charge rate
    'lambda': 1e-2,             # dual function weight
    'priceRise': 0.,            # amount (as a decimal percent) which the price rises when all cars are being charged
    'beta': 1.0,                # exponentiate price by beta [CURRENTLY UNUSED]
    'mu_stCharge':0.4,          # mean on starting charge level
    'std_stCharge':0.1,         # standard deviation on starting charge level
    'mu_arr':19,                # mean on arrival time
    'std_arr':2,                # standard deviation on arrival time
    'mu_dep':31,                # mean on departure time
    'std_dep':1,                # standard deviation on departure time
    'terminalFunc': (lambda x: -cp.inv_pos(x)),    # terminal reward function in terms of CVXPY atoms ### MUST BE SOCP-compatible for GUROBI
    'terminalFuncCalc': (lambda x: -1. / x),
}

def meanField(params=None,gmm=None,plotAgainstBase=False):

    if params is None:
        params = DEFAULT_PARAMS
    
    if gmm is None:
        gmm = makeModel(timeRange=params['timeRange'])

    eGMM = energyGMM(gmm)

    actions = int((params['timeRange'][-1]-params['timeRange'][0])/params['actRate'])
    cars = params['cars']
    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,eGMM.mle()) # normalize to make comparable on log scale

    # If beta is 1 and the price rise rate is 0, mf problem is solvable for each car individually
    if params['beta']==1. and params['priceRise']==0.:
        
        a = cp.Variable(actions,boolean=True)
        constraints = []
        for i,t in enumerate(x):
            if t < params['mu_arr'] or t >= params['mu_dep']:
                constraints += [a[i]==0]

        reward = -params['lambda']*params['actRate']*P.T*a + params['terminalFunc'](params['mu_stCharge']+params['actRate']*params['chargeRate']*cp.sum(a))
        prob = cp.Problem(cp.Maximize(reward),constraints)

        pstar = prob.solve()
        policy = np.zeros((cars,actions))
        policy[:,:] = a.value.round()
        reward = backtrackDualReward(pstar,P,policy,params)

        
    else:
        relaxation = False
        constraints = []
        if not relaxation:
            a = cp.Variable((cars,actions),boolean=True)
        else:
            params['cars'] = 1
            cars = params['cars']
            a = cp.Variable((cars,actions))
            constraints += [a>=0]
            constraints += [a<=1]

        for i,t in enumerate(x):
            if t < params['mu_arr'] or t >= params['mu_dep']:
                for c in range(cars):
                    constraints += [a[c,i]==0]
        for c in range(cars):
            constraints+=[params['mu_stCharge']+params['actRate']*params['chargeRate']*cp.sum(a[c,:]) <= 1.0]
        
        # calculate reward
        
        # electricity costs
        elec = 0
        fact = params['actRate']/float(cars)
        for i,t in enumerate(x):
            elec -= cp.sum(a[:,i])    *P[i]*cars
            elec -= cp.quad_form(a[:,i],np.ones((cars,cars)))*P[i]*params['priceRise']
        elec = elec*fact

        # final charge costs
        terminal = 0
        for i in range(cars):
            charge_i = params['mu_stCharge']
            for j in range(actions):
                charge_i += params['actRate']*params['chargeRate']*a[i,j]
            terminal += params['terminalFunc'](charge_i)

        total = params['lambda']*elec + terminal
        prob = cp.Problem(cp.Maximize(total),constraints)
        if not relaxation:
            pstar = prob.solve(solver=cp.GUROBI, verbose=True,TimeLimit=10.)
        else:
            pstar = prob.solve(solver=cp.SCS, verbose=True, max_iters=20000)
        
        policy = a.value

        #iters = prob.solver_stats.num_iters
        #solveTime = prob.solver_stats.solve_time

        reward = backtrackDualReward(pstar,P,policy,params)
    
    if plotAgainstBase:
        # Plot Charge Levels
        charges = meanFieldChargeFromPolicy(policy[0,:],params)
        plotChargeAgainstPrice(x,P,charges)

    return policy, reward

def backtrackDualReward(optimalReward,P,policy,params):
    dualWeight=params['lambda']
    actRate = params['actRate']
    if params['beta']==1. and params['priceRise']==0.:
        electricity = -P.T.dot(policy[0,:])
        finalCharge = optimalReward-dualWeight*actRate*electricity
        reward = {'totalReward':optimalReward, 'electricCosts':electricity,'finalCharge':finalCharge}
    else:
        numCarsCharging=carsCharging(policy)
        realPrice = P + params['priceRise']/params['cars']*P*numCarsCharging
        
        electricTimeCosts = realPrice*numCarsCharging
        electricity = -policy.dot(realPrice)
        finalCharge = params['terminalFuncCalc'](params['mu_stCharge']+params['actRate']*params['chargeRate']*policy.sum(axis=1))
        
        tol = 1e-2
        #assert abs(dualWeight*actRate*np.sum(electricity) + np.sum(finalCharge) - optimalReward) < tol
        
        reward = {'totalReward':optimalReward, 'electricCosts':electricity.mean(),'finalCharge':finalCharge.mean()}
        reward['finalChargeMin'] = finalCharge.min()
        reward['finalChargeMax'] = finalCharge.max()
        reward['finalChargeStd'] = finalCharge.std()
        reward['electricityMin'] = electricity.min()
        reward['electricityMax'] = electricity.max()
        reward['electricityStd'] = electricity.std()
        reward['realElectricPrice'] = realPrice
        reward['averageRewardPerCar'] = optimalReward / params['cars']
        reward['numCarsCharging'] = numCarsCharging
    return reward

def carsCharging(policy):
    return policy.sum(axis=0)

def MLEGMM(gmm):
    return gmm.means_.T.dot(gmm.weights_)


def monteCarloPolicy():
    pass

def plotChargeAgainstPrice(time,price,charges):
    plt.figure()
    plt.plot(time,price)
    plt.plot(time,charges.T)
    plt.xlabel('Simulation Time (hr)')
    plt.ylabel('Normalized Electricity Price, Charge State')
    plt.show()

def meanFieldChargeFromPolicy(policies,params):
    charges = np.zeros(policies.shape)
    charges[:,0] = params['mu_stCharge']
    for i in range(1,charges.shape[1]):
        charges[:,i] = charges[:,i-1] + params['actRate']*params['chargeRate']*policies[:,i-1]
    return charges

def noPriceChangeMeanFieldPareto():

    params = DEFAULT_PARAMS
    gmm = makeModel(timeRange=params['timeRange'])
    eGMM = energyGMM(gmm)

    # Get policies 
    lambdas = [0.,.25e-2,.5e-2,.75e-2,1e-2, 1.25e-2, 1.5e-2, 1.75e-2, 2e-2, .25e-1]
    rewards = np.zeros((2,len(lambdas)))
    policies = np.zeros((len(lambdas),80))
    for i, lam in enumerate(lambdas):
        print(lam)
        params['lambda'] = lam 
        policy, reward = meanField(gmm=gmm,params=params)
        rewards[0,i] = reward['electricCosts']
        rewards[1,i] = reward['finalCharge']
        policies[i,:] = policy[1,:]

    # Plot Pareto Front
    plt.figure()
    plt.plot(rewards[0,:], rewards[1,:])
    plt.title('Pareto Front for Mean Field Problem with no price changing')
    plt.xlabel('Electricity Costs')
    plt.ylabel('log(Final Car Charge)')
    
    # Plot Policies

    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,eGMM.mle()) # normalize to make comparable on log scale

    plt.figure()
    plt.plot(x,P)
    plt.plot(x,policies.T)
    plt.legend('Normalized Electricity Cost', 'Charge Policies')
    
    # Plot Charge Levels
    charges = meanFieldChargeFromPolicy(policies,params)
    plt.figure()
    plt.plot(x,P)
    plt.plot(x,charges.T)
    lambdalist = ['lambda=%05.2f' %(l) for l in lambdas]
    plt.legend(['Normalized Electricity Cost']+lambdalist)
    plt.title('Charge state over time for each lambda on pareto front')
    plt.xlabel('Simulation Time (hr)')
    plt.ylabel('Normalized Electricity Price, Charge State')

    plt.show()

def PriceChangeMeanFieldPareto():
    params = DEFAULT_PARAMS
    params['priceRise'] = 0.2 # 20% cost rise when all cars are being charged at once. Linear up to that point
    params['cars'] = 100
    gmm = makeModel(timeRange=params['timeRange'])
    eGMM = energyGMM(gmm)

    # Get policies 
    lambdas = [0.5e-1,0.8e-1,0.9e-1, 1.0e-1, 1.2e-1, 1.5e-1, 2.0e-1, 2.2e-1 ] # good for log, 20%
    #lambdas = [2e-2, 3e-2, 4e-2, 5e-2,6e-2,7e-2, 8e-2] # good for log, 100%
    rewards = np.zeros((2,len(lambdas)))
    policies = np.zeros((len(lambdas),80))
    prices = np.zeros((len(lambdas),80))
    for i, lam in enumerate(lambdas):
        print(lam)
        params['lambda'] = lam 
        policy, reward = meanField(gmm=gmm,params=params)
        rewards[0,i] = reward['electricCosts']
        rewards[1,i] = reward['finalCharge']
        policies[i,:] = reward['numCarsCharging']/params['cars']
        prices[i,:] = reward['realElectricPrice']

    # Plot Pareto Front
    plt.figure()
    plt.plot(rewards[0,:], rewards[1,:])
    plt.title('Pareto Front for Mean Field Problem with price changing')
    plt.xlabel('Electricity Costs')
    plt.ylabel('log(Final Car Charge)')
    
    # Plot Policies
    PmeanHourly = MLEGMM(gmm)
    meanPrice = PmeanHourly.mean()
    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,eGMM.mle()) # normalize to make comparable on log scale

    plt.figure()
    plt.plot(x,P)

    viridis = cm.get_cmap('viridis',len(lambdas))
    for column1,color in zip(prices,viridis.colors):
        plt.plot(x,column1,color=color)
    for column2,color in zip(policies,viridis.colors):
        plt.plot(x,column2,color=color)  
    lambdalist = ['lambda=%05.2f' %(l) for l in lambdas]
    plt.legend(['Normalized Electricity Cost']+lambdalist)
    plt.title('Percentage of cars charging and adjusted electricity cost over time')
    plt.xlabel('Time (hr)')
    plt.ylabel('Percentage cars charging, adjusted electricity cost')
    plt.show()

if __name__ == '__main__':

    #noPriceChangeMeanFieldPareto()
    PriceChangeMeanFieldPareto()
