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

def sgrad(w,xi,yi):
    return xi.T*((np.dot(xi,w)-yi))


def SGD(w_init,sgrad_method,eta):
    w=[w_init]
    w_last_check=w_init
    count=0
    iter_check=20
    N=Xbar.shape[0]
    for it in range(10):
        #shuffle data
        rd_id=np.random.permutation(N)
        for id in rd_id:
            count+=1
            xi= np.array(Xbar[id,:]).reshape(1,2)
            yi= np.array(y[id])
            g=sgrad_method(w[-1],xi,yi)
            w_new=w[-1]-eta*g
            w.append(w_new)
            if count%iter_check==0:
                w_this_check=w[-1]
                if np.linalg.norm(w_this_check-w_last_check)/len(w_init)<1e-3:
                    return w
                w_last_check=w_this_check
        return w


if __name__=='__main__':
    w_init=np.array([[1],[1]])
    w=SGD(w_init,sgrad,0.1)
    print('result: {0}, intern: {1}'.format(w[-1],len(w)))
    