import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt
from dataAggregator import *


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
    'terminalFunc': (lambda x: cp.log(cp.minimum(1,x))),    # terminal reward function in terms of CVXPY atoms
}

def meanField(params=None,gmm=None,plotAgainstBase=False):

    if params is None:
        params = DEFAULT_PARAMS
    
    if gmm is None:
        gmm = makeModel(timeRange=params['timeRange'])

    PmeanHourly = MLEGMM(gmm)
    meanPrice = PmeanHourly.mean() 

    actions = int((params['timeRange'][-1]-params['timeRange'][0])/params['actRate'])
    cars = params['cars']
    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,PmeanHourly)/meanPrice # normalize to make comparable on log scale

    # If beta is 1 and the price rise rate is 0, mf problem is solvable for each car individually
    if params['beta']==1. and params['priceRise']==0.:
        
        a = cp.Variable(actions,boolean=True)
        constraints = []
        for i,t in enumerate(x):
            if t < params['mu_arr'] or t >= params['mu_dep']:
                constraints += [a[i]==0]

        reward = -params['lambda']*P.T*a + params['terminalFunc'](params['mu_stCharge']+params['actRate']*params['chargeRate']*cp.sum(a))
        prob = cp.Problem(cp.Maximize(reward),constraints)

        pstar = prob.solve()
        policy = np.zeros((cars,actions))
        policy[:,:] = a.value.round()
        reward = backtrackDualReward(pstar,P,policy,params)

        
    else:
        a = cp.Variable((cars,actions),boolean=True)
        constraints = []
        for i,t in enumerate(x):
            if t < params['mu_arr'] or t >= params['mu_dep']:
                constraints += [a[:,i]==0]
        
        # calculate reward
        elec = 0
        
        # electricity costs
        for i,t in enumerate(x):
            elec -= P[i]*cp.sum(a[:,i]) + P[i]*params['priceRise']/cars*cp.sum(a[:,i])**2

        # final charge costs
        logcharge = 0
        for i in range(cars):
            logcharge += params['terminalFunc'](params['mu_stCharge']+params['actRate']*params['chargeRate']*cp.sum(a[i,:]))
        
        prob = cp.Problem(cp.Maximize(elec*params['lambda']+logcharge),constraints)
        pstar = prob.solve()
        policy = a.value

        reward = 0 #backtrackDualReward(pstar,P,policy,params)
    
    if plotAgainstBase:
        # Plot Charge Levels
        charges = meanFieldChargeFromPolicy(policy[0,:],params)
        plotChargeAgainstPrice(x,P,charges)

    return policy, reward

def backtrackDualReward(optimalReward,P,policy,params):
    dualWeight=params['lambda']
    
    if params['beta']==1. and params['priceRise']==0.:
        electricity = -P.T.dot(policy[0,:])
        finalCharge = optimalReward-dualWeight*electricity
        reward = {'totalReward':optimalReward, 'electricCosts':electricity,'finalCharge':finalCharge}
    else:
        electricity = -P.T.dot(policy)
        finalCharge = optimalReward-dualWeight*electricity
        


        
        tol = 1e-3
        assert dualWeight*np.sum(electricity) + np.sum(finalCharge) - optimalReward < tol
        
        reward = {'totalReward':optimalReward, 'electricCosts':electricity.mean(),'finalCharge':finalCharge.mean()}
        reward['finalChargeMin'] = np.min(finalCharge)
        reward['finalChargeMax'] = np.max(finalCharge)
        reward['finalChargeStd'] = np.std(finalCharge)
        reward['electricityMin'] = np.min(electricity)
        reward['electricityMax'] = np.max(electricity)
        reward['electricityStd'] = np.std(electricity)
        reward['averageRewardPerCar']=optimalReward / params['cars']
    return reward



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
    
    # Get policies 
    lambdas = [1e-2,2e-2,3e-2,4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 1e-1]
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
    PmeanHourly = MLEGMM(gmm)
    meanPrice = PmeanHourly.mean()
    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,PmeanHourly)/meanPrice # normalize to make comparable on log scale

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
    params['priceRise'] = 0.33 # 33% cost rise when all cars are being charged at once. Linear up to that point
    gmm = makeModel(timeRange=params['timeRange'])
    
    # Get policies 
    lambdas = [1e-2] # ,2e-2,3e-2,4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 1e-1]
    rewards = np.zeros((2,len(lambdas)))
    policies = np.zeros((len(lambdas),80))
    for i, lam in enumerate(lambdas):
        print(lam)
        params['lambda'] = lam 
        policy, reward = meanField(gmm=gmm,params=params)
        #rewards[0,i] = reward['electricCosts']
        #rewards[1,i] = reward['finalCharge']
        #policies[i,:] = policy[1,:]


if __name__ == '__main__':

    noPriceChangeMeanFieldPareto()
    #PriceChangeMeanFieldPareto()
