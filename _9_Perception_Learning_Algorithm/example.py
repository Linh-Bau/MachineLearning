import numpy as np
import random
import matplotlib.pyplot as plt
#-2x+y+6=0

x1=np.random.uniform(3,10,100)
y1=np.zeros_like(x1)
for i in range(100):
    y1[i]=2*x1[i]-6-random.uniform(1,3)

x2=np.random.uniform(1,10,100)
y2=np.zeros_like(x2)
for i in range(100):
    y2[i]=2*x2[i]-6+random.uniform(1,3)



plt.scatter(x1,y1,color='r')
plt.scatter(x2,y2,color='b')


plt.show()