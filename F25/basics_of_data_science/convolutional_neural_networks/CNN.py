import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# -----------------------
# Utility functions
# -----------------------
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

# -----------------------
# Convolution layer
# -----------------------
class Conv2D:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))

    def forward(self, X):

        '''

        CREATE YOUR OWN FORWARD FUNCTION

        '''

        return self.out

    def backward(self, d_out, lr=0.01):
        batch_size, depth, h, w = self.X.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        dX = np.zeros_like(self.X)
        out_h, out_w = self.out.shape[2], self.out.shape[3]

        for i in range(out_h):
            for j in range(out_w):
                region = self.X[:, :, i:i+self.filter_size, j:j+self.filter_size]
                for f in range(self.num_filters):
                    d_filters[f] += np.sum(region * d_out[:, f, i, j][:, None, None, None], axis=0)
                    d_biases[f] += np.sum(d_out[:, f, i, j])
                    dX[:, :, i:i+self.filter_size, j:j+self.filter_size] += self.filters[f] * d_out[:, f, i, j][:, None, None, None]

        self.filters -= lr * d_filters / batch_size
        self.biases -= lr * d_biases / batch_size
        return dX

# -----------------------
# Max Pool layer
# -----------------------
class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, X):

        '''

        CREATE YOUR OWN FORWARD FUNCTION

        '''

        return self.out

    def backward(self, d_out):
        dX = np.zeros_like(self.X)
        batch, depth, h, w = self.X.shape
        out_h, out_w = d_out.shape[2], d_out.shape[3]
        for i in range(out_h):
            for j in range(out_w):
                dX[:, :, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size] += \
                    self.mask[:, :, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size] * d_out[:, :, i, j][:, :, None, None]
        return dX

# -----------------------
# Fully connected layer
# -----------------------
class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))

    def forward(self, X):

        '''

        CREATE YOUR OWN FORWARD FUNCTION

        '''

        return X @ self.W + self.b

    def backward(self, d_out, lr=0.01):
        dW = self.X.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        dX = d_out @ self.W.T
        self.W -= lr * dW / len(self.X)
        self.b -= lr * db / len(self.X)
        return dX

# -----------------------
# Load MNIST via sklearn
# -----------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalize and reshape to (N, 1, 28, 28)
X = (X / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)
y = y.astype(int)

# Split into train and test sets
train_X, test_X = X[:60000], X[60000:]
train_y, test_y = y[:60000], y[60000:]

train_y_onehot = one_hot(train_y, 10)
test_y_onehot = one_hot(test_y, 10)

# Use subset for faster training
train_X = train_X[:2000]
train_y = train_y[:2000]
train_y_onehot = train_y_onehot[:2000]

# -----------------------
# Build CNN model
# -----------------------
conv = Conv2D(num_filters=8, filter_size=3, input_depth=1)
pool = MaxPool2D(size=2)
fc = Dense(input_dim=8*13*13, output_dim=10)

# -----------------------
# Training loop
# -----------------------
lr = 0.05
epochs = 50
batch_size = 4

for epoch in range(epochs):
    idx = np.random.permutation(len(train_X))
    train_X, train_y, train_y_onehot = train_X[idx], train_y[idx], train_y_onehot[idx]

    losses, accuracies = [], []
    for start in range(0, len(train_X), batch_size):
        end = start + batch_size
        X_batch = train_X[start:end]
        y_batch = train_y_onehot[start:end]
        y_labels = train_y[start:end]

        '''

        CREATE YOUR OWN FORWARD AND LOSS DESCRIPTIONS

        '''

        # Forward


        # Loss & Accuracy


        # Backward
        d_logits = (probs - y_batch) / len(X_batch)
        d_fc = fc.backward(d_logits, lr=lr)
        d_pool = d_fc.reshape(out.shape)
        d_pool = pool.backward(d_pool)
        d_relu = d_pool * relu_grad(conv.out)
        conv.backward(d_relu, lr=lr)

    print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Accuracy={np.mean(accuracies):.4f}")

# -----------------------
# Testing
# -----------------------
def evaluate(X, y_onehot, y_labels):
    out = conv.forward(X)
    out = relu(out)
    out = pool.forward(out)
    out_flat = out.reshape(len(X), -1)
    logits = fc.forward(out_flat)
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y_labels)
    return predictions, accuracy

preds, test_acc = evaluate(test_X[:1000], test_y_onehot[:1000], test_y[:1000])
print(f"\nTest Accuracy on 1000 samples: {test_acc:.4f}")

# -----------------------
# Visualization
# -----------------------
def visualize_predictions(X, y_labels, num_samples=5):
    samples = np.random.choice(len(X), num_samples, replace=False)
    images = X[samples]
    labels = y_labels[samples]

    preds, _ = evaluate(images, one_hot(labels, 10), labels)

    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[idx,0], cmap='gray')
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"Pred: {preds[i]}\nTrue: {labels[i]}", color=color)
        plt.axis('off')
    plt.show()

# Show predictions
visualize_predictions(test_X, test_y)
