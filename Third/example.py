# %reset
import numpy as np 
from mnist import MNIST # require `pip install python-mnist`
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from display_network import *

from display_network import *

#load dữ liệu
mndata = MNIST('./MNIST/') # path to your MNIST folder 
mndata.load_testing()

#ma trận x thu được là ma trận
X = np.array(mndata.test_images)

#phân loại
K=10
kmeans=KMeans(n_clusters=K,random_state=0).fit(X)
pred_labels=kmeans.predict(X)


#lấy ramdom 20 bức ảnh ứng với mỗi label
M=None
for k in range(0,10):
    k_cluster=X[pred_labels==k,:]
    ramdom_index=np.random.choice(k_cluster.shape[0],1,replace=False)
    k_cluster_data=k_cluster[ramdom_index,:].ravel()
    if M is None:
        M= k_cluster_data
    else:
        M= np.vstack((M, k_cluster[ramdom_index,:].ravel().T))

S=np.vstack((kmeans.cluster_centers_.T,M.T))


A=display_network(S,10,2)

f1 = plt.imshow(A, interpolation='nearest', cmap = "gray")

f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()

