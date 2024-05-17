import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=2))


def knn_prediction(X_train,y_train,X_test,k):
    distances = euclidean_distance(X_train, X_test)
    nearest_indices = np.argsort(distances,axis=0)[:k]
    nearest_labels = y_train[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, axis=0, return_counts=True)
    return unique_labels[np.argmax(counts)]

def knn(X_train, y_train, X_test, k):
    predictions = []
    for x_test in X_test:
        predictions.append(knn_prediction(X_train, y_train, x_test, k))
    return np.array(predictions)



X, y = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

errors = []
for k in range(1, int(np.sqrt(len(X_train)))):
    y_pred = knn(X_train, y_train, X_test, k)
    error = np.mean(y_pred != y_test)
    errors.append(error)




fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x', s=100)
axes[0].set_title('Scatter Plot of Random Dataset')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, int(np.sqrt(len(X_train)))), errors, marker='o')
axes[1].set_title('Error Rate vs. k')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Error Rate')
axes[1].set_xticks(range(1, int(np.sqrt(len(X_train)))))
axes[1].grid(True)

plt.tight_layout()
plt.show()