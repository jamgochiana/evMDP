import numpy as np 
import matplotlib.pyplot as plt

mean = 50
std = 10

times = np.arange(mean-4*std, mean+4*std+1)
charge_constraints = np.zeros(times.shape)


for i,t in enumerate(times):
    draws = np.random.normal(loc=mean, scale=std, size=(10000,))
    charge_constraints[i] = np.maximum(0,draws-t).mean()
stop_time = times[np.where(charge_constraints<1.)][0]
print(stop_time)
print((stop_time-mean)/std)

plt.figure()
plt.plot(times, charge_constraints)
plt.plot(times, np.ones(times.shape))
plt.show()
