import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, learning_rate=0.01, num_iterations=50000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(num_features, 2) * 0.2
        self.parameters['b1'] = np.zeros((1, 2))
        self.parameters['W2'] = np.random.randn(2, 1) * 0.2
        self.parameters['b2'] = np.zeros((1, 1))

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.parameters['W1']) + self.parameters['b1']
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.parameters['W2']) + self.parameters['b2']
        A2 = self.sigmoid(Z2)
        return A1, A2

    def backward_propagation(self, X, y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.parameters['W2'].T) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.parameters['W1'] -= self.learning_rate * dW1
        self.parameters['b1'] -= self.learning_rate * db1
        self.parameters['W2'] -= self.learning_rate * dW2
        self.parameters['b2'] -= self.learning_rate * db2

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_parameters(n)

        for i in range(self.num_iterations):
            # Forward propagation
            A1, A2 = self.forward_propagation(X)

            # Compute loss
            loss = (-1 / m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
            self.losses.append(loss)

            # Backward propagation
            dW1, db1, dW2, db2 = self.backward_propagation(X, y, A1, A2)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)

    def predict(self, X):
        A1, A2 = self.forward_propagation(X)
        predictions = np.round(A2)
        return predictions

# Example usage:
# Read X.txt
with open('DeepLearningPlayground/X.txt', 'r') as file:
    lines = file.readlines()
    X_data = []
    for line in lines:
        numbers = line.strip().split()
        row = [int(num) for num in numbers]
        X_data.append(row)

# Convert X_data to a NumPy array
X_train = np.array(X_data)

# Read y.txt
with open('DeepLearningPlayground/y.txt', 'r') as file:
    lines = file.readlines()
    y_data = []
    for line in lines:
        number = int(line.strip())
        y_data.append([number])

# Convert y_data to a NumPy array
y_train = np.array(y_data)
X_train2 = np.array([[0, 0], [2, 1], [6, 3], [4, 2], [7, 8], [3, 1], [9, 6], [7, 1], [5, 3], [4, 8], [7, 8], [9, 7], [6, 5], [2, 9], [9, 3], [4, 7], [2, 5], [8, 2]])
y_train2 = np.array([[1], [1], [1], [1], [0], [1], [0], [1], [1], [0], [0], [0], [0], [0], [0], [0], [1], [1]])
model = DeepNeuralNetwork()
model.fit(X_train, y_train)

X_test = np.array([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [9, 10], [8, 10], [8, 9], [7, 10], [7, 9], [7, 8], [6, 10], [6, 9], [6, 8], [6, 7], [5, 10], [5, 9], [5, 8], [5, 7], [5, 6], [4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [3, 10], [3, 9], [3, 8], [3, 7], [3, 6], [3, 5], [3, 4], [2, 10], [2, 9], [2, 8], [2, 7], [2, 6], [2, 5], [2, 4], [2, 3], [1, 10], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1]])
predictions = model.predict(X_test)
predictions = predictions.flatten()

plt.scatter(X_test[predictions==0.0][:,0], X_test[predictions==0.0][:,1], color='yellow')
plt.scatter(X_test[predictions==1.0][:,0], X_test[predictions==1.0][:,1], color='green')
plt.show()
plt.plot(range(len(model.losses)), model.losses)
plt.show()