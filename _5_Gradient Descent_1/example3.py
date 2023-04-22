import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

def grad(w):
    N=Xbar.shape[0]
    return 1/N*Xbar.T.dot(Xbar.dot(w)-y)

def cost(w):
    N=Xbar.shape[0]
    return 0.5/N*np.linalg.norm(y-Xbar.dot(w),2)**2

def myGD1(eta,w0):
    w=[w0]
    for it in range(100):
        w_new=w[-1] -eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w,it)

if __name__=='__main__':
    (x1,it1)= myGD1(1,np.array([[2],[1]]))
    print('Solution x1 = ',x1[-1], 'cost = ',cost(x1[-1]), 'obtained after ',it1,' iterations')
    plt.plot(np.array(x1)[:,0],np.array(x1)[:,1],color='r')
    plt.axis('on')
    plt.show()

    pass
