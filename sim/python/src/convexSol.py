import cvxpy as cp
import numpy as np 
import matplotlib.pyplot as plt
from dataAggregator import *


DEFAULT_PARAMS = {
    'cars': 100,
    'timeRange': [14, 34],
    'actRate': 1./4, 
    'chargeRate': 1./5,
    'lambda': 1e-2,
    'priceRise': 0.,
    'beta': 1.0,
    'mu_stCharge':0.4,
    'std_stCharge':0.1,
    'mu_arr':19,
    'std_arr':2,
    'mu_dep':31,
    'std_dep':1,
    'terminalFunc': (lambda x: cp.log(cp.minimum(1,x))),
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
        policy[:,:] = a.value.astype('int8')

        def backtrackDualReward(optimalReward,P,actions,dualWeight):
            electricity = -P.T.dot(actions)
            finalCharge = optimalReward-dualWeight*electricity
            reward = {'totalReward':optimalReward, 'electricCosts':electricity,'finalCharge':finalCharge}
            return reward

        reward = backtrackDualReward(pstar,P,a.value,params['lambda'])

        if plotAgainstBase:
            plotActionsAgainstPrice(x,P,a.value)

        return policy, reward

    print('Not yet implemented')
    pass

def MLEGMM(gmm):
    return gmm.means_.T.dot(gmm.weights_)


def monteCarloPolicy():
    pass

def plotActionsAgainstPrice(time,price,actions):
    plt.figure()
    plt.plot(time,[price, actions])
    plt.show()

def noPriceChangeMeanFieldPareto():

    params = DEFAULT_PARAMS
    gmm = makeModel(timeRange=params['timeRange'])
    
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    rewards = np.zeros((2,len(lambdas)))
    policies = np.zeros((len(lambdas),80))
    for i, lam in enumerate(lambdas):
        print(lam)
        params['lambda'] = lam 
        policy, reward = meanField(gmm=gmm,params=params)
        rewards[0,i] = reward['electricCosts']
        rewards[1,i] = reward['finalCharge']
        policies[i,:] = policy[1,:]

    plt.figure()
    plt.plot(rewards[0,:], rewards[1,:])
    plt.title('Pareto Front for Mean Field Problem with no price changing')
    plt.xlabel('Electricity Costs')
    plt.ylabel('log(Final Car Charge)')
    
    PmeanHourly = MLEGMM(gmm)
    meanPrice = PmeanHourly.mean()
    xp = np.arange(params['timeRange'][0],params['timeRange'][-1]+1)
    x = np.arange(params['timeRange'][0],params['timeRange'][-1],params['actRate'])
    P = np.interp(x,xp,PmeanHourly)/meanPrice # normalize to make comparable on log scale
    
    plt.figure()
    plt.plot(x,P)
    plt.plot(x,policies[2])
    plt.legend('Normalized Electricity Cost', 'Charge Policies')
    plt.show()

if __name__ == '__main__':

    noPriceChangeMeanFieldPareto()
    
