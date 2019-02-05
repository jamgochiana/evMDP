import math
import numpy as np
import ev
import matplotlib.pyplot as plt
## Define potential transition models for how cars arrive/depart. 
## Should include uncertainty information

def Residential(state,model_params):
    # In the residential model, we have sigmoid CDFs dictating 
    # arrival and departure times. 
    
    # If passed empty model parameters, save some default distributions
    if 'arr_mean' not in model_params.keys():
        model_params['arr_mean'] = 19.
        model_params['arr_std'] = 2.
        model_params['dep_mean'] = 31.
        model_params['dep_std'] = 1.
        model_params['arrival_cutoff'] = 25.
    # Must reset seed every time
    
    base_seed = model_params['base_seed']
    charges = state.charges.copy()
    
    for iCar in range(len(charges)):
        
        # Car Arrival Modeling
        arrival_prob = GaussianCDF(state.t,model_params['arr_mean'], model_params['arr_std'])
        if charges[iCar] == -1. and state.t < model_params['arrival_cutoff']:
            
            # set seed for arriving car 
            np.random.seed(len(charges)*base_seed+iCar)
            
            #make comparison
            if np.random.rand() < arrival_prob:
                charges[iCar] = newCarCharge(model_params)
            continue
        
        # Car Departure Modeling
        departure_prob = GaussianCDF(state.t,model_params['dep_mean'], model_params['dep_std'])
        if charges[iCar] >= 0.0:
            
            # set seed for departing car 
            np.random.seed(len(charges)*base_seed+len(charges)+iCar)
            
            # make comparison
            if np.random.rand() < departure_prob:
                charges[iCar] = -1.
            continue
        
    return ev.evState(state.t, charges)

# Gaussian Cumulative Distribution Function          
def GaussianCDF(x,mu,sig):
    return 0.5*(1+math.erf((x-mu)/(sig*math.sqrt(2))))

# Charge level for arriving car
def newCarCharge(model_params):
    return 0.25

if __name__=='__main__':
    # build unit test
    model_params = {'base_seed': 2020}
    n = 10000
    t = [14,34]
    start_charges = np.zeros(n) - 1.
    state = ev.evState(t[0],start_charges)
    states = []
    carsPresent = []
    times = np.linspace(t[0],t[1],4*(t[1]-t[0])+1)
    for t in times:
        state.t = t
        state = Residential(state,model_params)
        states.append(state)
        carsPresent.append(np.count_nonzero(state.charges != -1.)/n)
    
    p1=plt.plot(times, np.array(carsPresent))
    p2=plt.plot(times[:-1], np.diff(np.array(carsPresent))/np.diff(times))
    plt.show()