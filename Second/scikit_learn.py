import numpy as np
from sklearn.cluster import KMeans

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

k_means=KMeans(n_clusters=K,random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(k_means.cluster_centers_)