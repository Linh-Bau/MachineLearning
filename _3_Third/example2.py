import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from convert import *

img=mpimg.imread('./girl3.jpg')

# plt.imshow(img)
# imgplot=plt.imshow(img)
# plt.axis('off')
# plt.show()

X=img.reshape((img.shape[0]*img.shape[1],img.shape[2]))

K=3
kmeans=KMeans(n_clusters=K,random_state=0).fit(X)
center_cluster=kmeans.cluster_centers_
labels=kmeans.labels_
#thay giá trị của mỗi pixel bằng center cluster chứa nó
for k in range(K):
    X[labels==k,:]=center_cluster[k]
    pass


#save result to a file
np.savez('example2',kmeans_center=center_cluster,kmeans_label=labels)

plt.imshow(img)
plt.axis('off')
plt.show()

