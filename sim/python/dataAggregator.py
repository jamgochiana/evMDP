import os

import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import mixture

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def processData(folder):
    df = aggregateDataFiles(folder)
    df = cleanData(df)
    
    df = getWinterWeekdays(df)
    #gpFun(df)
    mixtureModel(df)
    plt.show()
    
    return df

def aggregateDataFiles(folder):
    try:
        filelist = os.listdir(folder)
    except:
        print("Invalid Directory")
        raise
    df_all = pd.DataFrame()
    #print("[%s]" % (" " * len(filelist)))
    #print(" ",end='')
    for file in filelist:
        read_file = os.path.join(folder,file)
        df = readFile(read_file)
        df_all = df_all.append(df,ignore_index=True)
        #print("#",end='')
    #print(" ")
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
    
#    # Plot LML landscape
#    plt.figure(3)
#    theta0 = np.logspace(-2, 3, 49)
#    theta1 = np.logspace(-2, 0, 50)
#    Theta0, Theta1 = np.meshgrid(theta0, theta1)
#    LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
#            for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
#    LML = np.array(LML).T
#    
#    vmin, vmax = (-LML).min(), (-LML).max()
#    vmax = 50
#    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
#    plt.contour(Theta0, Theta1, -LML,
#                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
#    plt.colorbar()
#    plt.xscale("log")
#    plt.yscale("log")
#    plt.xlabel("Length-scale")
#    plt.ylabel("Noise-level")
#    plt.title("Log-marginal-likelihood")
#    plt.tight_layout()
    
  


def mixtureModel(df):
    data = df.pivot(index='Date', columns='Hour', values='Demand')
    data = data.dropna()
    X = data.values
    
    def plotBaseline(hours,gmm0):
        plt.plot(hours, gmm0.means_[0,:], 'k', lw=3, zorder=9)
        plt.fill_between(hours, gmm0.means_[0] - np.sqrt(np.diag(gmm0.covariances_[0])),
                     gmm0.means_[0] + np.sqrt(np.diag(gmm0.covariances_[0])),
                     alpha=0.5, color='k')
    
    def baseline(X,plot=True):
        gmm0 = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X)
        hours = np.arange(24)
        
        if plot:
            plt.figure()
            plotBaseline(hours,gmm0)
            plt.plot(hours,X.T,'b')
            plt.title('Demand Data')
        return gmm0
    
    gmm0 = baseline(X)
    
    hours = np.arange(24)
    for n in range(1,6):
        gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(X)
        demand_sampled, label_sampled = gmm.sample(n_samples=25)
        
        plt.figure()
        plotBaseline(hours,gmm0)
        plt.plot(hours,demand_sampled.T)
        plt.title('20 samples using %02d Gaussians to fit Demand Data' %(n))

def getWinterWeekdays(df):
    ww = df[df['Month']% 11 <= 2]
    ww = ww[ww['Weekday Num'] < 5]
    return ww

if __name__ == "__main__":
    foldername = './data/CISO_ConsumptionData/'
    df = processData(foldername)