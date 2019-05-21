import sys
sys.path.append("src")

import numpy as np
import cvxpy as cp
from energyGMM import *
from evChargeState import *
from evMPC import *
from dataAggregator import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# make base demand model
tr = [19, 32]
gmm = makeModel(timeRange=tr)
eGMM = energyGMM(gmm, time_range=tr)

cars = 100
rise = 0.2/cars
dt=0.25
t0 = tr[0]
tf = 31
mpc = evMPC(eGMM,cars,t=t0, rise=rise, end_times=tf)

baseline = False
if baseline:
    sample = eGMM.mle()
    outname = "baselineMPC_%04dcars.gif" %(cars)
else:
    seed_ = 1999
    np.random.seed(seed_)
    sample = eGMM.sample()[0,:]
    outname = "stochMPCseed%04d_%04dcars.gif" %(seed_,cars)


fig, ax = plt.subplots()
ln0, = plt.plot(mpc.demand_model.time_range,sample)
ln1, = plt.plot([],[])
ln2, = plt.plot([],[])
ln3, = plt.plot([],[])
ln4, = plt.plot([],[])
ln5, = plt.plot([],[])
ax.set_xlabel('Time (hr)')
ax.set_ylabel('Cars Charging (%), Demand')
fig.suptitle('Past/Predicted Charging w/ MPC')

if not baseline:
    MLE = mpc.demand_model.posterior_mle()
    STD = mpc.demand_model.posterior_std()
    ln5.set_data(mpc.demand_model.time_range,MLE) 
    ax.fill_between(mpc.demand_model.time_range, MLE-STD, MLE+STD, 
        alpha=0.5, color='b')

def update(i):
    mpc.step(sample[i], dt)
    past_demand, future_demand = mpc.total_demand()
    past_ratio, future_ratio = mpc.charging_ratio()

    ln1.set_data(*past_demand)
    ln2.set_data(*future_demand)
    ln3.set_data(*past_ratio)
    ln4.set_data(*future_ratio)

    if not baseline:
        MLE = mpc.demand_model.posterior_mle()
        STD = mpc.demand_model.posterior_std()
        ln5.set_ydata(MLE) 
        ax.collections.clear()
        ax.fill_between(mpc.demand_model.time_range, MLE-STD, MLE+STD, 
        alpha=0.5, color='b')

    return ln1, ln2, ln3, ln4, ln5, ax

def init():
    ax.set_ylim(0, sample.max()*(1+rise*cars))
    return ax

anim = FuncAnimation(fig,update,frames=(tf-t0), init_func=init, interval=1000)
anim.save("pics/%s" %(outname), dpi=80, writer='imagemagick')


"""
while len(mpc.times_remaining) > 0:
    mpc.step(baseline[i], dt)
    past_demand, future_demand = mpc.total_demand()
    past_ratio, future_ratio = mpc.charging_ratio()

    plt.figure()
    plt.plot(*past_demand)
    plt.plot(*future_demand)
    plt.plot(*past_ratio)
    plt.plot(*future_ratio)
    i+=1

plt.show()
"""