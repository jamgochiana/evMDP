import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def RealTime(t, modelGP = None):
    if modelGP is None:
        
        # fit real time-cost function from data, build GP model
        pass
    
    cost = 0
    # sample from GP model
    return cost

# returns price per kilowatt at time of use
def TimeOfUse(t,model_name):
    
    # different time of use costs
    
    if model_name == "PG&E Summer":
    # https://www.pge.com/en_US/small-medium-business/your-account/rates-and-rate-options/time-of-use-rates.page
        times = np.array([0.,8.5,12.,18.,21.5])
        costs = np.array([0.219, 0.246, 0.270, 0.246, 0.219])
    
    elif model_name == "PG&E Summer Peak Day":
    # https://www.pge.com/en_US/small-medium-business/your-account/rates-and-rate-options/peak-day-pricing.page
        times = np.array([0.,8.5,12.,18.,21.5])
        costs = np.array([0.209, 0.236, 0.260, 0.236, 0.209])
        
    elif model_name == "PG&E Winter":
    # https://www.pge.com/en_US/small-medium-business/your-account/rates-and-rate-options/peak-day-pricing.page
        times = np.array([0.,8.5,21.5])
        costs = np.array([0.206, 0.227, 0.206])
    
    time = t % 24.
    
    idx = (time>=times).nonzero()[0][-1]
    return costs[idx]

# obtain costs from leaving cars
def getTerminalCost(s,a,sp):
    reward = 0
    for i in range(len(s.charges)):
        if sp.charges[i] == -1 and s.charges[1] != -1:
            
            finalcharge = s.charges[i]
            
            # re-work final charge state
            #if a[i] == 1:
            #    
            #    finalcharge += 
        
            reward += np.log(finalcharge)
                
    return reward