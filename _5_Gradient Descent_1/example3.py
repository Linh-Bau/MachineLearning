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
    _N=Xbar.shape[0]
    return 1/_N*Xbar.T.dot(Xbar.dot(w)-y)

def cost(w):
    _N=Xbar.shape[0]
    return 0.5/_N*np.linalg.norm(y-Xbar.dot(w),2)**2

def myGD1(eta,w0):
    _w=[w0]
    for _it in range(100):
        _w_new=_w[-1] -eta*grad(_w[-1])
        if np.linalg.norm(grad(_w_new))/len(_w_new) < 1e-3:
            break
        _w.append(_w_new)
    return (_w,_it)


if __name__=='__main__':
    (_x1,_it1)= myGD1(1,np.array([[2],[1]]))
    print('Solution x1 = ',_x1[-1], 'cost = ',cost(_x1[-1]), 'obtained after ',_it1,' iterations')
    #reshape ma trận
    _x1=np.array(_x1)
    _x1=np.reshape(_x1,(_x1.shape[0],_x1.shape[1]))
    _x1_x=_x1[:,0]
    _x1_y=_x1[:,1]
   

    _x=np.arange(0,10,.1)
    _y=np.arange(0,10,.1)
    _X,_Y=np.meshgrid(_x,_y)
    _Z = np.zeros(_X.shape)
    _levels=np.linspace(0,5,50)
    for i in range(_X.shape[0]):
        for j in range(_X.shape[1]):
            _xij=_X[i,j]
            _yij=_Y[i,j]
            _w=grad(np.array([[_xij], [_yij]]))
            # print(_w)
            # print('--------------')
            _Z[i,j] = np.linalg.norm(_w)
    
    print(_Z)
    cmap = plt.get_cmap('rainbow')
    plt.contour(_X,_Y,_Z,cmap=cmap,levels=_levels)
    plt.plot(4,3,'go')
    plt.xlim(0,5)
    plt.ylim(0,5)
    plt.plot(_x1_x,_x1_y,'r-')
    plt.show() 

   