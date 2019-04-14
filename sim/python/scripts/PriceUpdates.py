import sys
sys.path.append("src")
import os, shutil
import dataAggregator
from energyGMM import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

tr = [14, 34]
time = np.arange(tr[0], tr[1]+1)
seed_ = 1999
try:
    os.mkdir("pics")
except:
    pass

seedpath = "pics/seed%04d" %(seed_)
try: 
    shutil.rmtree(seedpath)
except:
    pass
os.mkdir(seedpath)

np.random.seed(seed_)

gmm = dataAggregator.makeModel(timeRange=tr,mixtures=[5])

eGMM = energyGMM(gmm, time_range=tr)
MLE = eGMM.mle()
STD = eGMM.std()


"""
samples_old = gmm.sample(n_samples=50)[0]
samples_new = eGMM.sample(n_samples=50)

plt.figure()
plt.plot(time,MLE, 'k', lw=3, zorder=9)
plt.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
plt.plot(time,samples_old.T / gmm.means_.T.dot(gmm.weights_).mean())

plt.figure()
plt.plot(time,MLE, 'k', lw=3, zorder=9)
plt.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
plt.plot(time,samples_new.T)

plt.show()
"""

sample = eGMM.sample(seed=seed_) 

#plt.figure()
#plt.plot(time,MLE, 'k', lw=3, zorder=9)
#plt.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
#plt.plot(time,sample.T)
#plt.savefig("pics/seed%04d/PriceUpdate00.png" %(seed_))

fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(time,MLE, 'k', lw=3, zorder=9)
ax.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
ax.plot(time,sample.T,'r')

line1, = ax.plot(time, MLE, 'b', lw=3, zorder=9)
ax.fill_between(time, MLE - 0.0001*STD, MLE + 0.0001*STD, alpha=0.5, color='b')

#for i, t in enumerate(time[:-1]):
def update(i):    
    eGMM.observe(sample[0,i])
    MLE2 = eGMM.posterior_mle()
    STD2 = eGMM.posterior_std()

    #plt.figure()
    #plt.plot(time,MLE, 'k', lw=3, zorder=9)
    #plt.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
    #plt.plot(time,sample.T,'r')
    #plt.plot(time,MLE2, 'b', lw=3, zorder=9)
    #plt.fill_between(time, MLE2 - STD2, MLE2 + STD2, alpha=0.2, color='b')
    #plt.savefig("pics/seed%04d/PriceUpdate%02d.png" %(seed_,i+1))
    line1.set_ydata(MLE2)
    ax.collections.clear()
    ax.fill_between(time, MLE - STD, MLE + STD, alpha=0.5, color='k')
    ax.fill_between(time, MLE2 - STD2, MLE2 + STD2, alpha=0.2, color='b')
    return line1, ax

# must add custom init function, otherwise FuncAnimation will do first index twice
def init():
    pass

anim = FuncAnimation(fig,update,frames=len(time)-1, init_func=init, interval=1000)
anim.save("pics/seed%04d/PriceSim.gif" %(seed_), dpi=80, writer='imagemagick')

