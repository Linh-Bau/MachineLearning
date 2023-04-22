from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_display(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
#khởi tạo các centers ban đầu.
def kmeans_init_centers(X,k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


#Gán nhán mới cho các điểm khi biết các centers.
def kmeans_assign_labels(X,centers):
    #tính toán khoảng cách giữa các điểm ở ma trận x và centers
    D=cdist(X,centers) #kết quả trả về sẽ là 1 ma trận 
    return np.argmin(D,axis=1)
    #Trong NumPy, hàm np.argmin() được sử dụng để tìm chỉ số của phần tử có giá trị nhỏ nhất trong một mảng đa chiều.
    #Tham số axis được sử dụng để chỉ định trục trên đó hàm sẽ hoạt động. Khi axis = 1, hàm sẽ tìm chỉ số của phần tử nhỏ nhất trên mỗi hàng của mảng.
    #Ví dụ, nếu D là một ma trận 2 chiều, thì np.argmin(D, axis=1) sẽ trả về một mảng 1 chiều có độ dài bằng với số hàng của D, 
    #trong đó phần tử thứ i sẽ là chỉ số của phần tử nhỏ nhất trên hàng thứ i của D.

def kmeans_update_centers(X,labels,K):
    centers=np.zeros((K,X.shape[1])) # ma trận Kxn
    for k in range(K):
          # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers


def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])