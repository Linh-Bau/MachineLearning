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
X_bar = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

#thêm nhiễu
X1=np.concatenate((X1,[[2],[2]]),axis=1)
X_bar=np.concatenate((X_bar,[[1],[2],[2]]),axis=1)
y=np.concatenate((y,[[-1]]),axis=1)

#region Gradient Descent
def grad(xi,yi):
    return -np.dot(xi,yi.T)


def assign_labels(w,data):
    labels=np.zeros((1,data.shape[1]))
    for i in range(labels.shape[1]):
        if np.dot(w.T,data[:,i])>=0:
            labels[0,i]=1
        else:
            labels[0,i]=-1
    return labels

def GD(w_init,eta,X_bar,labels):
    w=[w_init]
    for it in range(100):
        cal_labels=assign_labels(w[-1], X_bar)
        missing_index_r,missing_index_c=np.where(cal_labels!=labels)
        if len(missing_index_c)==0: #tất cả các điểm đã sắp xếp ok
            return w
        xi=X_bar[:,missing_index_c]
        yi=labels[0,missing_index_c].reshape((1,len(missing_index_c)))
        g=grad(xi,yi)
        w_new=w[-1]-eta*g
        w.append(w_new)
    return w
#endregion

#region GD with momentom
def GD_momentum(w_init,eta,gamma,X_bar,labels):
    w=[w_init]
    v_old=np.zeros_like(w_init)
    for it in range(100):
        cal_labels=assign_labels(w[-1], X_bar)
        missing_index_r,missing_index_c=np.where(cal_labels!=labels)
        if len(missing_index_c)==0: #tất cả các điểm đã sắp xếp ok
            return w
        xi=X_bar[:,missing_index_c]
        yi=labels[0,missing_index_c].reshape((1,len(missing_index_c)))
        g=grad(xi,yi)
        v_new=gamma*v_old+eta*g
        w_new=w[-1]-v_new
        v_old=v_new
        w.append(w_new)
#endregion

#region GD with Nesterov accelerated gradient
#đạo hàm hàm mất mát k phụ thuộc theta nên cái này k áp dụng được
#endregion

#region Stochatic Gradient Descent (SGD)

def perceptron(w_init,X_bar,labels):
    w=[w_init]
    epoches=50
    N=X_bar.shape[1]
    for epoch in range(epoches):
        rd_indexs=np.random.permutation(N)
        cal_labels=assign_labels(w[-1],X_bar)
        if np.array_equal(cal_labels,labels):
            return w
        for index in rd_indexs:
            if cal_labels[0,index]==labels[0,index]:
                continue
            xi=X_bar[:,index].reshape(3,1)
            yi=labels[0,index]
            w_new=w[-1]+yi*xi
            w.append(w_new)
    return w

#endregion



if __name__=='__main__':
    w_init=np.array([[2],[2],[2]])
    #w=GD(w_init,0.1,X_bar,y)
    #w=GD_momentum(w_init,0.1,1,X_bar,y)
    w=perceptron(w_init,X_bar,y)
    print(f'w= {w[-1]}, intern: {len(w)}')
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
            title.set_text(f'intern: {frame+1}, w0= {w[frame][0][0]:.2f}, w1= {w[frame][1][0]:.2f}, w2={w[frame][2][0]:.2f}')
        return _2d_line,title
    ani=FuncAnimation(fig,update,500,interval=50,blit=False, repeat=False)
    plt.show()
