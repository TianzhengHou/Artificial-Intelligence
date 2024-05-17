import numpy as np
import matplotlib.pyplot as plt

center1 = (10, 10)
center2 = (0, 0)
center3 = (10, -5)

cov1 = [[5, 1], [1, 5]]
cov2 = [[10, 2], [2, 10]]
cov3 = [[3, 1], [1, 3]]

data1 = np.random.multivariate_normal(center1, cov1, size=500)
data2 = np.random.multivariate_normal(center2, cov2, size=500)
data3 = np.random.multivariate_normal(center3, cov3, size=500)

x1, y1 = data1[:, 0], data1[:, 1]
x2, y2 = data2[:, 0], data2[:, 1]
x3, y3 = data3[:,0], data3[:,1]


def kmean(X,k,max_iter = 100):
    centroids = X[np.random.choice(X.shape[0],k,replace=False)]
    for _ in range(max_iter):
        # np.newaxis is used to add a new dimension to X
        # np.linalg.norm calculates the distance between every point in X and the centroids
        distance = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) #(n_sample, k)
        # Find the index of the smallest distance
        labels = np.argmin(distance,axis=1)
        # re-assign the centroid to the new position by taken the mean of distance and apply for all centroids.
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        # break if the new centroid is equal to the old centroids to speed up the process
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels

X = np.concatenate([data1, data2, data3], axis=0)
k = 3
centroids, labels = kmean(X, k)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(k):
    points = X[labels == i]
    plt.scatter(points[:, 0], points[:, 1], color=colors[i])

plt.show()