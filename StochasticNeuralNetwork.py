import numpy as np

from sklearn.datasets import load_iris


#data prep
iris = load_iris()
X, y = iris.data, iris.target

# z-score Normalization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std

# One-hot encode the target variable
n_samples, n_features = X_normalized.shape
n_classes = len(np.unique(y))
y_one_hot = np.eye(n_classes)[y]

# Split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.2):
    test_size = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_one_hot, test_size=0.2)

#Class object
class Layer:
    def __init__(self,n_input,n_neurons):
        #random init weight and biases
        self.weight = 0.10 * np.random.randn(n_input,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weight) + self.biases

    def backward(self, input, output_gradient, learning_rate):
        #update weight and biases based on gradient and learning rate
        input_gradient = np.dot(output_gradient, self.weight.T)
        weight_gradient = np.dot(input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weight -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        
        return input_gradient

class Activation_Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))

    def backward(self, output_gradient):
        return output_gradient * (self.output * (1 - self.output))

class Activation_Softmax:
    def forward(self, input):
        #output layer final activition function
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#Create a network that have 3 layers and 5 nods in each layer expect the input
input_layer = Layer(n_features, 5)
hidden_layer = Layer(5, 5)
output_layer = Layer(5, n_classes)
sigmoid = Activation_Sigmoid()
softmax = Activation_Softmax()

epochs = 10000
learning_rate = 0.01
batch_size = 32

for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    total_loss = 0

    for i in range(0,X_train_shuffled.shape[0],batch_size):
        # Forward pass
        input_layer.forward(X_train_shuffled[i:i+batch_size])
        sigmoid.forward(input_layer.output)
        hidden_layer.forward(sigmoid.output)
        sigmoid.forward(hidden_layer.output)
        output_layer.forward(sigmoid.output)
        softmax.forward(output_layer.output)

        # MSE loss function
        loss = np.mean((softmax.output - y_train_shuffled[i:i+batch_size]) ** 2)
        total_loss += loss

        # Backward pass
        output_gradient = 2 * (softmax.output - y_train_shuffled[i:i+batch_size]) # Derivative of the loss function
        hidden_gradient = output_layer.backward(hidden_layer.output, output_gradient, learning_rate)
        hidden_activation_gradient = sigmoid.backward(hidden_gradient)
        input_gradient = hidden_layer.backward(input_layer.output, hidden_activation_gradient, learning_rate)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss/X_train_shuffled.shape[0]}')

#Test
input_layer.forward(X_test)
sigmoid.forward(input_layer.output)
hidden_layer.forward(sigmoid.output)
sigmoid.forward(hidden_layer.output)
output_layer.forward(sigmoid.output)
softmax.forward(output_layer.output)

predictions = np.argmax(softmax.output, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)
print("Test Accuracy:", accuracy)