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

def mgrad(w,rd_i):
    xi=Xbar[rd_i,:]
    yi=y[rd_i]
    return xi.T.dot((np.dot(xi,w)-yi))


def MGD(w_init,sgrad_method,eta):
    w=[w_init]
    w_last_check=w_init
    count=0
    iter_check=10
    N=Xbar.shape[0]
    n=1000
    for it in range(10):
        #shuffle data
        rd_id=np.random.permutation(N)
        g=sgrad_method(w[-1],rd_id[0:n])
        w_new=w[-1]-eta*g
        w.append(w_new)
        w_this_check=w[-1]
        if np.linalg.norm(w_this_check-w_last_check)/len(w_init)<1e-3:
            return w
        w_last_check=w_this_check
    return w


if __name__=='__main__':
    w_init=np.array([[1],[1]])
    w=MGD(w_init,mgrad,0.1)
    print('result: {0}, intern: {1}'.format(w[-1],len(w)))
    