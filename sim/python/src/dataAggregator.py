import os

import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import mixture

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def processData(folder, filters=[]):
    df = aggregateDataFiles(folder)
    df = cleanData(df)
    
    for filterFunc, filterArgs in filters:
        df = filterFunc(df,*filterArgs)

    return df

"""

"""
def fitData(X, mixtures=range(1,6), hours=list(range(24)), plot=False):
    gmms = mixtureModel(X,mixtures,hours,plot)
    
    if len(mixtures)==1:
        gmms = gmms[0]
    return gmms

def aggregateDataFiles(folder):
    try:
        filelist = os.listdir(os.path.join(os.getcwd(),folder))
    except:
        print("Invalid Directory")
        raise
    df_all = pd.DataFrame()
    
    tenpercent = len(filelist)//10
    print(len(filelist))
    for iFile, file in enumerate(filelist):
        read_file = os.path.join(folder,file)
        df = readFile(read_file)
        df_all = df_all.append(df,ignore_index=True)
        if (iFile+1) % tenpercent == 0:
            print("%04.1d%% Done Reading Files" %(100.*(iFile+1.)/len(filelist)))
    return df_all
    
    
def readFile(filename):
    return pd.read_csv(filename, header= 5, names = ['Date','Demand', 'Forecast'], engine='python')

def cleanData(df):
    
    dates =  pd.to_datetime(df['Date'])
    df['Date Time'] = dates
    df['Date'] = dates.dt.date
    df['Hour'] = dates.dt.hour
    df['Weekday Name'] = dates.dt.weekday_name
    df['Weekday Num'] = dates.dt.weekday
    df['Month'] = dates.dt.month
    df = df.drop_duplicates(subset = 'Date Time')
    df = df.dropna()
    df = df.sort_values(by='Date Time').reset_index(drop = True)
    
    
    return df

def gpFun(df):
    # pull data for winter weekdays
    X = df['Hour'].values
    y = df['Demand'].values
    
    print('Here')
    X = X[:,np.newaxis]
    #y = y[:,np.newaxis]
    
    plt.figure(0)
    plt.plot(X[:,0],y,'b')
    
    # First run
    plt.figure(1)
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e4, 1e8))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True).fit(X, y)
    X_ = np.linspace(0, 24, 100)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    
    plt.scatter(X[:, 0], y, c='r', s=20, zorder=10, edgecolors=(0, 0, 0))
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)), \
                     y_mean + np.sqrt(np.diag(y_cov)), \
                     alpha=0.5, color='k')
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()
    
    # Second run
    plt.figure(2)
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e5, noise_level_bounds=(1e4, 1e8))
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0,normalize_y=True).fit(X, y)
    X_ = np.linspace(0,24, 100)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    
    plt.scatter(X[:, 0], y, c='r', s=20, zorder=10, edgecolors=(0, 0, 0))
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()


    plt.figure(3)
    y_samples = gp.sample_y(X_[:, np.newaxis], 5)
    
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.plot(X_, y_samples, lw=1)
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()
    
    plt.figure(4)

    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.tight_layout()


def dataFromFrame(df,hours=None):
    
    # Get frame
    data = df.pivot(index='Date', columns='Hour', values='Demand')
    data = data.dropna()
    X = (data.values).copy()
    if hours is not None:
        # temporary solution, not technically continuous stream from data but hopefully good enough 
        X = reformatHours(X,hours)
    return X

def reformatHours(X,hours):
    newIndices = hours % 24 
    Xp = X[:,newIndices]
    return Xp


def mixtureModel(X, mixtures=range(1,6), hours=list(range(24)), plot=False):

    gmm = [None]*len(mixtures) # preallocate
    for iGMM, n in enumerate(mixtures):
        gmm[iGMM] = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(X)
        
        if plot:
            samples = 25
            demand_sampled, label_sampled = gmm[iGMM].sample(n_samples=samples)
        
            plt.figure()
            plotBaselineBackground(hours,gmm[0])
            plt.plot(hours,demand_sampled.T)
            plt.title('%02d samples using %02d Gaussians to fit Demand Data' %(samples,n))
    
    if plot:
        plt.show()
    return gmm

def getWinterWeekdays(df):
    ww = df[df['Month']% 11 <= 2]
    ww = ww[ww['Weekday Num'] < 5]
    return ww

def timeRange2Hours(timeRange):
    return np.arange(timeRange[0],timeRange[-1]+1)

def plotBaselineBackground(hours,gmm0):
        plt.plot(hours, gmm0.means_[0,:], 'k', lw=3, zorder=9)
        plt.fill_between(hours, gmm0.means_[0] - np.sqrt(np.diag(gmm0.covariances_[0])),
                     gmm0.means_[0] + np.sqrt(np.diag(gmm0.covariances_[0])),
                     alpha=0.5, color='k')


def plotData(X,hours=np.arange(24),gmm0=None):
    plt.figure()
    if gmm0 is not None:
        plotBaselineBackground(hours,gmm0)
    plt.plot(hours,X.T,'b')
    plt.title('Demand Data')
    plt.show()



if __name__ == "__main__":
    foldername = './data/CISO_ConsumptionData/'
    
    timeRange = [18,36]
    hours = timeRange2Hours(timeRange)
    # filters = [(filterTime,(hours,)),]

    df = processData(foldername) #,filters)
    X = dataFromFrame(df,hours)
    print(X.shape)
    gmm = fitData(X, range(1,5), hours, plot=True)
    
    plotData(X,gmm0=gmm[0])