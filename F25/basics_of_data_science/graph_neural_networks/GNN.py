import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from urllib.request import urlretrieve

# ---------------------------
# Utility functions
# ---------------------------

def load_cora():
    urlretrieve("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", "cora.tgz")
    import tarfile
    with tarfile.open("cora.tgz", "r:gz") as tar:
        tar.extractall()
    data = np.genfromtxt("cora/cora.content", dtype=str)
    features = np.array(data[:, 1:-1], dtype=float)
    labels_raw = data[:, -1]
    classes = list(set(labels_raw))
    labels = np.array([classes.index(lbl) for lbl in labels_raw])
    node_ids = data[:, 0]
    node_idx = {node: i for i, node in enumerate(node_ids)}

    edges_unordered = np.genfromtxt("cora/cora.cites", dtype=str)
    edges = np.array([[node_idx[edge[0]], node_idx[edge[1]]] for edge in edges_unordered])

    adj = np.zeros((len(labels), len(labels)))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1

    return features, labels, adj

# Normalization of adjacency matrix
def normalize_adjacency(A):
    I = np.eye(A.shape[0])
    A_hat = A + I
    D = np.array(A_hat.sum(1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy(preds, labels):
    m = preds.shape[0]
    log_likelihood = -np.log(preds[range(m), labels] + 1e-9)
    return np.sum(log_likelihood) / m

# One-hot encoding
def one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1
    return out

# ---------------------------
# GCN Layer and Model
# ---------------------------
class GCNLayer:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.dW = np.zeros_like(self.W)

    def forward(self, X, A_hat):



        return self.Z

    def backward(self, dZ, lr):



        return dX

class GCN:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, X, A_hat):



        return self.A2

    def backward(self, X, A_hat, labels, lr):





# ---------------------------
# Visualization functions
# ---------------------------

def visualize_embeddings(features, labels, title="Node Embeddings Visualization"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        plt.scatter(reduced[labels == c, 0], reduced[labels == c, 1], label=f"Class {c}", s=20)
    plt.legend()
    plt.title(title)
    plt.show()

def visualize_graph(A, node_colors, labels=None, title='Graph Visualization'):
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    unique_colors = np.unique(node_colors)
    cmap = plt.cm.get_cmap('tab10', len(unique_colors))
    for cls in unique_colors:
        nodes = [n for n in range(len(node_colors)) if node_colors[n] == cls]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[cmap(cls)], label=f'Class {cls}', node_size=80)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.show()

# ---------------------------
# Training and evaluation
# ---------------------------
features, labels, A = load_cora()
A_hat = normalize_adjacency(A)
num_nodes, in_dim = features.shape
hidden_dim = 16
out_dim = len(np.unique(labels))

# Train/test split
np.random.seed(42)
indices = np.arange(num_nodes)
np.random.shuffle(indices)
train_idx = indices[:140]
val_idx = indices[140:640]
test_idx = indices[1708:]

model = GCN(in_dim, hidden_dim, out_dim)

# Training
lr = 0.01
epochs = 200
for epoch in range(epochs):
    preds = model.forward(features, A_hat)
    loss = cross_entropy(preds[train_idx], labels[train_idx])
    model.backward(features, A_hat, labels, lr)
    if epoch % 20 == 0:
        val_loss = cross_entropy(preds[val_idx], labels[val_idx])
        val_acc = np.mean(np.argmax(preds[val_idx], axis=1) == labels[val_idx])
        print(f"Epoch {epoch}: Loss={loss:.4f}, Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}")

# Evaluation
preds = model.forward(features, A_hat)
pred_labels = np.argmax(preds, axis=1)
test_acc = np.mean(pred_labels[test_idx] == labels[test_idx])
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize embeddings
visualize_embeddings(preds, labels, title="GCN Learned Node Embeddings")

# Visualize graph colored by predicted labels
visualize_graph(A, pred_labels, labels, title="GCN Predictions on Cora Graph")

# Visualize graph colored by actual labels
visualize_graph(A, labels, labels, title="Cora Graph with True Labels")

# Visualize graph highlighting mispredictions (0=correct,1=wrong)
diff = (pred_labels != labels).astype(int)
visualize_graph(A, diff, labels, title="Cora Graph: Misclassifications (Red=Wrong, Blue=Correct)")
