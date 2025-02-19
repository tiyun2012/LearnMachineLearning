import numpy as np
from PIL import Image
import random

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
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        return self.a3  # Returns probabilities

    def backward(self, X, y_true, learning_rate):
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

        self.weights3 -= learning_rate * dw3
        self.bias3 -= learning_rate * db3
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1

    def train(self, X, y, num_epochs, learning_rate, batch_size):
        m = X.shape[0]
        for epoch in range(num_epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            if epoch % 10 == 0:
                loss = self.cross_entropy_loss(self.forward(X), y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)  # 0 for A, 1 for Unknown

    @staticmethod
    def test_static():
        print("Testing static")

    def test_method(self):
        print("Testing method")

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-15
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def augment_image_combined(self, image, num_images=100, rotation_range=(-30, 30),
                                 translation_ratio=1/3, scale_ratio=1/3):
        """
        Generate a list of num_images augmented images from a 32x32 PIL image.
        Each image is produced by applying a random scaling, then rotation,
        and finally a translation.
        """
        augmented_images = []
        width, height = image.size

        for _ in range(num_images):
            # Random scaling
            scale_factor = random.uniform(1 - scale_ratio, 1 + scale_ratio)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            scaled = image.resize(new_size, Image.BILINEAR)
            # If scaled image is smaller, pad it; if larger, center-crop it.
            if scaled.size[0] < width or scaled.size[1] < height:
                padded = Image.new("L", (width, height), color=255)
                x_offset = (width - scaled.size[0]) // 2
                y_offset = (height - scaled.size[1]) // 2
                padded.paste(scaled, (x_offset, y_offset))
                img1 = padded
            else:
                left = (scaled.size[0] - width) // 2
                top = (scaled.size[1] - height) // 2
                img1 = scaled.crop((left, top, left + width, top + height))

            # Random rotation
            angle = random.uniform(rotation_range[0], rotation_range[1])
            rotated = img1.rotate(angle, fillcolor=255)

            # Random translation
            max_dx = width * translation_ratio
            max_dy = height * translation_ratio
            dx = random.uniform(-max_dx, max_dx)
            dy = random.uniform(-max_dy, max_dy)
            translated = rotated.transform((width, height), Image.AFFINE,
                                           (1, 0, dx, 0, 1, dy), fillcolor=255)

            augmented_images.append(translated)

        return augmented_images

def run():
    # Simulated dataset for training (replace with real data)
    X_train = np.random.rand(1000, 1024)  # 1000 samples, 32x32 flattened
    y_train = np.random.randint(0, 2, size=(1000,))  # Labels: 0 (A), 1 (Unknown)

    def one_hot_encode(labels, num_classes):
        m = labels.shape[0]
        encoded = np.zeros((m, num_classes))
        encoded[np.arange(m), labels] = 1
        return encoded

    y_train_one_hot = one_hot_encode(y_train, 2)
    model = NeuralNetwork()
    model.train(X_train, y_train_one_hot, num_epochs=500, learning_rate=0.01, batch_size=32)
    predictions = model.predict(X_train[:10])
    print("Predictions:", predictions)
