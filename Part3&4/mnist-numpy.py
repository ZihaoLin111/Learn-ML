import numpy as np
import os
import struct


class Dataset:
    def __init__(self, images_path, labels_path):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)

    def load_images(self, path):
        with open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
        return images

    def load_labels(self, path):
        with open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_loss(X, y):
    Z2 = X
    S2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
    log_likelihood = -np.log(S2[np.arange(len(y)), y])
    loss = np.mean(log_likelihood)
    return loss


def get_accuracy(X, y):
    Z2 = X
    S2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
    predictions = np.argmax(S2, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


def train_epoch(X, y, W1, W2, learning_rate, batch_size):
    n = len(y)
    for i in range(0, n, batch_size):
        X_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        Z1 = np.dot(X_batch, W1)
        Z2 = np.dot(np.maximum(Z1, 0), W2)
        S2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
        S2[np.arange(len(y_batch)), y_batch] -= 1
        G2 = S2
        G1 = np.dot(G2, W2.T)
        G1[Z1 <= 0] = 0
        grad_W2 = np.dot(np.maximum(Z1, 0).T, G2) / len(y_batch)
        grad_W1 = np.dot(X_batch.T, G1) / len(y_batch)
        W2 -= learning_rate * grad_W2
        W1 -= learning_rate * grad_W1


def init_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(0)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    return W1, W2


def train(dataset, epochs, learning_rate, batch_size, hidden_dim, W1, W2):
    X = dataset.images / 255.0
    y = dataset.labels
    for epoch in range(epochs):
        train_epoch(X, y, W1, W2, learning_rate, batch_size)
        loss = get_loss(np.dot(np.maximum(np.dot(X, W1), 0), W2), y)
        acc = get_accuracy(np.dot(np.maximum(np.dot(X, W1), 0), W2), y)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")


def test(dataset, W1, W2):
    X = dataset.images / 255.0
    y = dataset.labels
    acc = get_accuracy(np.dot(np.maximum(np.dot(X, W1), 0), W2), y)
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    data_path = "./data/archive"

    train_images_path = os.path.join(data_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(data_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(data_path, "t10k-labels.idx1-ubyte")

    train_dataset = Dataset(train_images_path, train_labels_path)
    test_dataset = Dataset(test_images_path, test_labels_path)

    W1, W2 = init_weights(28 * 28, 128, 10)

    train(
        train_dataset,
        epochs=10,
        learning_rate=0.1,
        batch_size=64,
        hidden_dim=128,
        W1=W1,
        W2=W2,
    )
    test(test_dataset, W1, W2)
