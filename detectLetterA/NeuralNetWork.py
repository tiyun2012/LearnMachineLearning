import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=1024, hidden_size1=128, hidden_size2=64, output_size=2):
        # Initialize weights and biases
        np.random.seed(42)
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.bias1 = np.zeros((1, hidden_size1))
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.bias2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.bias3 = np.zeros((1, output_size))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        return self.a3  # Probabilities for each class

    def backward(self, X, y_true, learning_rate):
        # Backpropagation logic (same as before)
        m = X.shape[0]
        y_pred = self.a3

        dz3 = y_pred - y_true
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = np.dot(dz3, self.weights3.T) * self.relu_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.weights2.T) * self.relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights3 -= learning_rate * dw3
        self.bias3 -= learning_rate * db3
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
    def train(self, X, y, num_epochs, learning_rate, batch_size):
        m = X.shape[0]  # Total number of samples
        for epoch in range(num_epochs):
            # Shuffle the data
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, m, batch_size):
                # Create batches
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward and backward pass
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            if epoch % 10 == 0:  # Print loss every 10 epochs
                loss = self.cross_entropy_loss(self.forward(X), y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)  # Class index (0 for A, 1 for Unknown)
    @staticmethod
    def test_static():
        print("Testing static")

    def test_method(self):
        print("Testing method")


def run():
    # Simulated dataset (replace with real data)
    X_train = np.random.rand(1000, 1024)  # 1000 samples, 32x32 flattened
    y_train = np.random.randint(0, 2, size=(1000,))  # Labels: 0 (A), 1 (Unknown)

    # Convert labels to one-hot encoding
    def one_hot_encode(labels, num_classes):
        m = labels.shape[0]
        encoded = np.zeros((m, num_classes))
        encoded[np.arange(m), labels] = 1
        return encoded

    y_train_one_hot = one_hot_encode(y_train, 2)

    # Train the model
    model = NeuralNetwork()
    model.train(X_train, y_train_one_hot, num_epochs=500, learning_rate=0.01)

    # Predict on test data
    predictions = model.predict(X_train[:10])
    print("Predictions:", predictions)
