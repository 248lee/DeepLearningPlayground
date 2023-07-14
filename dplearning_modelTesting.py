import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    def __init__(self, learning_rate=0.03, num_iterations=30000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(num_features, 3) * 2
        self.parameters['b1'] = np.zeros((1, 3))
        self.parameters['W2'] = np.random.randn(3, 3) * 2
        self.parameters['b2'] = np.zeros((1, 3))
        self.parameters['W3'] = np.random.randn(3, 3) * 2
        self.parameters['b3'] = np.zeros((1, 3))
        self.parameters['W4'] = np.random.randn(3, 1) * 2
        self.parameters['b4'] = np.zeros((1, 1))

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.parameters['W1']) + self.parameters['b1']
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.parameters['W2']) + self.parameters['b2']
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(A2, self.parameters['W3']) + self.parameters['b3']
        A3 = self.sigmoid(Z3)
        Z4 = np.dot(A3, self.parameters['W4']) + self.parameters['b4']
        A4 = self.sigmoid(Z4)
        return A1, A2, A3, A4

    def backward_propagation(self, X, y, A1, A2, A3, A4):
        m = X.shape[0]
        dZ4 = A4 - y
        dW4 = (1 / m) * np.dot(A3.T, dZ4)
        db4 = (1 / m) * np.sum(dZ4, axis=0, keepdims=True)
        dZ3 = np.dot(dZ4, self.parameters['W4'].T) * (A3 * (1 - A3))
        dW3 = (1 / m) * np.dot(A2.T, dZ3)
        db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.dot(dZ3, self.parameters['W3'].T) * (A2 * (1 - A2))
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.parameters['W2'].T) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2, dW3, db3, dW4, db4

    def update_parameters(self, dW1, db1, dW2, db2, dW3, db3, dW4, db4):
        self.parameters['W1'] -= self.learning_rate * dW1
        self.parameters['b1'] -= self.learning_rate * db1
        self.parameters['W2'] -= self.learning_rate * dW2
        self.parameters['b2'] -= self.learning_rate * db2
        self.parameters['W3'] -= self.learning_rate * dW3
        self.parameters['b3'] -= self.learning_rate * db3
        self.parameters['W4'] -= self.learning_rate * dW4
        self.parameters['b4'] -= self.learning_rate * db4

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_parameters(n)

        for i in range(self.num_iterations):
            # Forward propagation
            A1, A2, A3, A4 = self.forward_propagation(X)

            # Compute loss
            loss = (-1 / m) * np.sum(y * np.log(A4) + (1 - y) * np.log(1 - A4))
            self.losses.append(loss)

            # Backward propagation
            dW1, db1, dW2, db2, dW3, db3, dW4, db4 = self.backward_propagation(X, y, A1, A2, A3, A4)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2, dW3, db3, dW4, db4)

    def predict(self, X):
        A1, A2, A3, A4 = self.forward_propagation(X)
        predictions = np.round(A4)
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
X_train = np.array(X_data)

# Read y.txt
with open('DeepLearningPlayground/y.txt', 'r') as file:
    lines = file.readlines()
    y_data = []
    for line in lines:
        number = int(line.strip())
        y_data.append([number])
y_train = np.array(y_data)

X_train2 = np.array([[0, 0], [2, 1], [6, 3], [4, 2], [7, 8], [3, 1], [9, 6], [7, 1], [5, 3], [4, 8], [7, 8], [9, 7], [6, 5], [2, 9], [9, 3], [4, 7], [2, 5], [8, 2]])
y_train2 = np.array([[1], [1], [1], [1], [0], [1], [0], [1], [1], [0], [0], [0], [0], [0], [0], [0], [1], [1]])
model = DeepNeuralNetwork()
model.fit(X_train, y_train)

# Read X_test.txt
with open('DeepLearningPlayground/X_test.txt', 'r') as file:
    lines = file.readlines()
    X_test_data = []
    for line in lines:
        numbers = line.strip().split()
        row = [int(num) for num in numbers]
        X_test_data.append(row)
X_test = np.array(X_test_data)

predictions = model.predict(X_test)
predictions = predictions.flatten()

plt.scatter(X_train[y_train.flatten()==0.0][:,0], X_train[y_train.flatten()==0.0][:,1], color='yellow')
plt.scatter(X_train[y_train.flatten()==1.0][:,0], X_train[y_train.flatten()==1.0][:,1], color='green')
plt.show()

plt.scatter(X_test[predictions==0.0][:,0], X_test[predictions==0.0][:,1], color='yellow')
plt.scatter(X_test[predictions==1.0][:,0], X_test[predictions==1.0][:,1], color='green')
plt.show()

plt.plot(range(len(model.losses)), model.losses)
plt.show()