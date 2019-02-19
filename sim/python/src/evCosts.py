import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd

def RealTime(t, modelGP = None):
    if modelGP is None:
        
        # fit real time-cost function from data, build GP model
        datafile = './data/NYISO-HourlyRealTime.csv'
        modelGP = fitGP(datafile)
    
    cost = 0
    # sample from GP model
    return cost

def fitGP(dataLoc):
    df = pd.read_csv(dataLoc)
    df[df.columns[1:]] = df[df.columns[1:]].replace('[\$,]','',regex=True).astype(float)
    dates =  pd.to_datetime(df.iloc[:,0])
    data = pd.DataFrame({'Date': dates.tolist(), 'Price': df.iloc[:,1].tolist()})
    data['Hour'] = data['Date'].dt.hour
    data['Weekday'] = data['Date'].dt.day_name
    data['Month'] = data['Date'].dt.month
    data.plot(x='Hour', y='Price')
    
    #data1 = data.pivot_table(index=data['Date'].dt.hour,values='Price')
    #data1.plot()
    a=1
    
    
    return -1
    
    
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

if __name__ == '__main__':
    datafile = './data/NYISO-HourlyRealTime.csv'
    GP = fitGP(datafile)
        