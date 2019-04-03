import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np

a = np.random.rand(50,12)
x = np.array([np.linspace(0,2*np.pi,num=50)])

c = np.array([np.linspace(0,10,12)])

y1 = a + np.cos(x).T + c
y2 = y1 + 10

viridis = cm.get_cmap('viridis',12)
for column,color in zip(y1.T,viridis.colors):
    plt.plot(x.T,column,color=color)

for column,color in zip(y2.T,viridis.colors):
    plt.plot(x.T,column,color=color)

plt.show()