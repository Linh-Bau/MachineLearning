import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(2)

X=np.random.rand(1000,1)
y=4+3*X+.2*np.random.rand(1000,1) #noise add

#Building Xbar
one=np.ones((X.shape[0],1))
Xbar=None
Xbar=np.concatenate((one,X),axis=1)

def mgrad(w,xi,yi):
    return xi.T.dot((np.dot(xi,w)-yi))


def MGD(w_init,sgrad_method,eta,Xbar,y):
    w=[w_init]
    batch_size=32
    epoches=100
    w_previous=w_init
    w_check_distance=20
    count=0
    #tính số lượng batch và thêm dữ liệu vào padding nếu cần
    n_batches=Xbar.shape[0] // batch_size
    if Xbar.shape[0] % batch_size !=0:
        n_batches+=1
        padding=batch_size-Xbar.shape[0]%batch_size
        Xbar=np.concatenate([Xbar,np.zeros((padding,Xbar.shape[1]))],axis=0)
        y=np.concatenate([y,np.zeros((padding,1))],axis=0)
    
    #huấn luyện mô hình
    for epoch in range(epoches):
        #xáo trộn các mẫu dữ liệu
        shuffle_indices=np.random.permutation(Xbar.shape[0])
        Xbar=Xbar[shuffle_indices]
        y=y[shuffle_indices]

        #lặp lại các batch và cập nhật trọng số
        for i in range(n_batches):
            count+=1
            batch_data=Xbar[i*batch_size:(i+1)*batch_size]
            batch_labels=y[i*batch_size:(i+1)*batch_size]
            #tính gradient trên batch hiện tại
            g=sgrad_method(w[-1],batch_data,batch_labels)
            w_new=w[-1]-eta*g/batch_size
            w.append(w_new)
            if count%w_check_distance==0:
                if np.linalg.norm(w[-1]-w_previous)/len(w_init)<1e-3:
                    return w
                w_previous=w[-1]
    return w


if __name__=='__main__':
    w_init=np.array([[1],[1]])
    w=MGD(w_init,mgrad,0.1,Xbar,y)
    print('result: {0}, intern: {1}'.format(w[-1],len(w)))
    