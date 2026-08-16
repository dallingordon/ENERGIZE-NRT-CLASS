import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Data generation
# ---------------------------
np.random.seed(42)

# Training data: sin with light noise
X_train = np.linspace(0, 10, 20).reshape(-1, 1)
y_train_true = np.sin(X_train).ravel()
train_noise = 0.2
y_train = y_train_true + train_noise * np.random.randn(len(X_train))

# Test data: cos with heavier noise
X_test = np.linspace(0, 10, 30).reshape(-1, 1)
y_test_true = np.cos(X_test).ravel()
test_noise = 0.5
y_test = y_test_true + test_noise * np.random.randn(len(X_test))

# Prediction grid
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)

# ---------------------------
# Step 2: Activation functions
# ---------------------------
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x) ** 2

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return sigmoid(x) * (1 - sigmoid(x))

activations = [
    ("ReLU", relu, relu_deriv),
    ("tanh", tanh, tanh_deriv),
    ("sigmoid", sigmoid, sigmoid_deriv)
]

# ---------------------------
# Step 3: Multi-layer Neural Network
# ---------------------------
class MultiLayerNN:
    def __init__(self, layer_sizes, activation, activation_deriv, lr=0.01):
        """
        layer_sizes: list of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.lr = lr
        self.activation = activation
        self.activation_deriv = activation_deriv

        # Weight initialization (small random values)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass through all layers."""
        self.zs = []
        self.as_ = [X]

        # Hidden layer(s)
        '''
        TO COMPLETE BY STUDENTS
        '''




        # Output layer (linear)
        '''
        TO COMPLETE BY STUDENTS
        '''





        self.zs.append(z)
        self.as_.append(z)
        return z

    def backward(self, y):
        """Backward pass with gradient descent updates."""
        m = len(y)
        y = y.reshape(-1, 1)

        # Gradient at output layer
        dz = (self.as_[-1] - y) / m
        for i in reversed(range(len(self.weights))):
            dW = self.as_[i].T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            if i > 0:  # propagate to previous layer
                da = dz @ self.weights[i].T
                dz = da * self.activation_deriv(self.zs[i-1])

            # Update weights and biases
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def train(self, X, y, epochs=5000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(y)

    def predict(self, X):
        return self.forward(X).ravel()

# ---------------------------
# Step 4: Train models with different activations
# ---------------------------
layer_sizes = [1, 100, 100, 1]  # input -> hidden1 -> hidden2 -> output
models = []
train_preds = []
test_preds = []

for name, act, act_deriv in activations:
    nn = MultiLayerNN(layer_sizes, act, act_deriv, lr=0.001)
    nn.train(X_train, y_train, epochs=100000)
    models.append((name, nn))

    # Predictions
    train_preds.append(nn.predict(X_plot))
    test_preds.append(nn.predict(X_test))

# ---------------------------
# Step 5: Plot results
# ---------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

# Training row
for i, (name, nn) in enumerate(models):
    ax = axes[0, i]
    ax.plot(X_plot, np.sin(X_plot), "k--", label="True function (sin)")
    ax.scatter(X_train, y_train, c="r", label="Train data")
    ax.plot(X_plot, train_preds[i], "b", label="NN prediction")
    ax.set_title(f"Training: {name}")
    ax.legend(loc="upper right")

# Testing row
for i, (name, nn) in enumerate(models):
    ax = axes[1, i]
    ax.plot(X_test, y_test_true, "k--", label="True function (cos)")
    ax.scatter(X_test, y_test, c="r", label="Test data")
    ax.plot(X_test, test_preds[i], "b", label="NN prediction")
    ax.set_title(f"Testing: {name}")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
