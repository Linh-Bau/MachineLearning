# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):    
    return np.array_equal(h(w, X), y) 

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi: # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 
                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)

print(w,m)
fig,ax=plt.subplots(1,1)
ax.scatter(X0[0,:],X0[1,:],color='g')
ax.scatter(X1[0,:],X1[1,:],color='b')

_2d_line, = ax.plot([],[],'r-')
title = ax.set_title('w=')

def update(frame):
    if frame<len(w):
        #wo,w1,w2
        #wo+w1*x+w2*y=0
        _y1=(w[frame][0][0]+w[frame][1][0]*1)/(-w[frame][2][0])
        _y2=(w[frame][0][0]+w[frame][1][0]*6)/(-w[frame][2][0])
        _2d_line.set_data([1,6],[_y1,_y2])
        title.set_text('w= {0}'.format(w[frame]))
    return _2d_line,title

ani=FuncAnimation(fig,update,100,interval=500,blit=False)

plt.show()