# %reset
import numpy as np 
from mnist import MNIST # require `pip install python-mnist`
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from display_network import *
from convert import *

#load dữ liệu
mndata = MNIST('./MNIST/') # path to your MNIST folder 
mndata.load_testing()

#ma trận x thu được là ma trận
X = np.array(mndata.test_images)

#phân loại
K=10
kmeans=KMeans(n_clusters=K,random_state=0).fit(X)
pred_labels=kmeans.predict(X)

M=None

#lấy ramdom 20 bức ảnh ứng với mỗi label
for k in range(0,10):
    k_cluster=X[pred_labels==k,:]
    ramdom_index=np.random.choice(k_cluster.shape[0],20,replace=False)
    k_cluster_data=k_cluster[ramdom_index,:].ravel()
    if M is None:
        M=k_cluster_data
    else:
        M= np.vstack((M, k_cluster_data))

S=np.hstack((kmeans.cluster_centers_,M))

A=convert_matrix(S,28,28)


f1 = plt.imshow(A, interpolation='nearest', cmap = "gray")

f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()

