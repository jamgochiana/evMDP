import os

import pandas as pd
import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import mixture

from matplotlib import pyplot as plt
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


def dataFromFrame(df):
    
    # Get frame
    data = df.pivot(index='Date', columns='Hour', values='Demand')
    data = data.dropna()
    X = (data.values).copy()
    return X

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

def filterTime(df,hours):
    assert hours[-1]-hours[0]<24
    # dataFromFrame uses the following (index='Date', columns='Hour', values='Demand')
    # we must therefore make sure the 'Hour' field lies in range of hours
    
    # Change hours
    shifted_hours = (df['Hour'].values - hours[0]) % 24 + hours[0]
    df['Hour'] = shifted_hours
    
    # Reduce day by (hours // 24) days
    day_shift = shifted_hours // 24
    shifted_dates = df['Date'].copy()
    for i, d in enumerate(day_shift.tolist()):
        shifted_dates[i] -= datetime.timedelta(d)
    df['Date'] = shifted_dates

    df = df[df['Hour']<=hours[-1]]
    return df

def makeModel(timeRange=[0,23], dataFolder='./data/CISO_ConsumptionData/', mixtures=[4], plotSamples=False, plotAllData=False):
    hours = timeRange2Hours(timeRange)
    filters = [(filterTime,(hours,)),]

    df = processData(dataFolder,filters)
    X = dataFromFrame(df)

    gmm = fitData(X, mixtures, hours, plotSamples)
    if plotAllData:
        plotData(X,hours,gmm0=gmm[0])

    if plotAllData or plotSamples:
        plt.show()
    
    return gmm    


if __name__ == "__main__":
    foldername = './data/CISO_ConsumptionData/'
    timeRange = [16,36]
    gmm = makeModel(timeRange,foldername,range(1,5), True, True)