import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(2)

X=np.random.rand(1000,1)
y=4+3*X+.2*np.random.rand(1000,1) #noise add

#Building Xbar
one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)

w_lr=np.dot(np.linalg.pinv(A),b)
print('solution found by fomular: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

