import random
import numpy as np
import matplotlib.pyplot as plt

def LR(X, t, lr, iterations):
    b = np.ones((len(X), 1))             # shape : (N, 1)
    X_b = np.append(X, b, axis=1)        # shape : (N, features + 1)
    theta = np.zeros((X_b.shape[1], 1))  # shape : (features + 1, 1)
    t = t.reshape(-1, 1)                 # shape : (N, 1)
    cost_list = []

    for _ in range(iterations):
        y = np.dot(X_b,theta)
        error = y - t
        gradient = np.dot(X_b.T, error)/len(X_b)
        theta -= lr*gradient
        cost = np.mean((error)**2)
        cost_list.append(cost)


    return cost_list, y, theta

def graph_lr(X, t, y, cost_list):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.title('Model Predictions vs Actual Targets')
    plt.scatter(X, t, c='g', s=10, label="targets")
    plt.scatter(X, y, c='b', s=10, label="predictions")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Model Cost')
    plt.plot(range(len(cost_list)), cost_list, c='b')
    plt.show()


X = 2 * np.random.rand(100, 1)
t = 4 + 3 * X + np.random.randn(100, 1)
lr = 0.01
iterations = 200
cost_list, y, theta = LR(X, t, lr, iterations)
graph_lr(X,t,y,cost_list)