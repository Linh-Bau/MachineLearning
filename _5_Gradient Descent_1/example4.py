import numpy as np
import matplotlib.pyplot as plt

def numerical_grad(w, cost):
    esp=1e-4
    g=np.zeros_like(w)
    for i in range(len(w)):
        w_p=w.copy()
        w_n=w.copy()
        w_p[i]+=esp
        w_n[i]-=esp
        g[i]=(cost(w_p)-cost(w_n))/(2*esp)
    return g

def check_grad(w,cost,grad):
    w=np.random.rand(w.shape[0],w.shape[1])
    grad1=grad(w)
    grad2=numerical_grad(w,cost)
    return True if np.linalg.norm(grad1-grad2)<1e-6 else False


if __name__=='__main__':
    np.random.seed(2)

    X = np.random.rand(1000, 1)
    y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

    # Building Xbar 
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X), axis = 1)

    def grad(w):
        _N=Xbar.shape[0]
        return 1/_N*Xbar.T.dot(Xbar.dot(w)-y)

    def cost(w):
        _N=Xbar.shape[0]
        return 0.5/_N*np.linalg.norm(y-Xbar.dot(w),2)**2

    isCheckOk=check_grad(np.random.rand(2,1),cost,grad)
    print('checking gradient...',isCheckOk)