import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Boltzmann Machine implementation
class BoltzmannMachine:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, batch_size=10):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialize weights and biases randomly
        self.weights = np.random.randn(num_visible, num_hidden)
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_visible(self, hidden_states):
        visible_activations = np.dot(hidden_states, self.weights.T) + self.visible_bias
        visible_probs = self.sigmoid(visible_activations)
        visible_states = (visible_probs > np.random.rand(self.num_visible)).astype(int)
        return visible_states

    def sample_hidden(self, visible_states):
        hidden_activations = np.dot(visible_states, self.weights) + self.hidden_bias
        hidden_probs = self.sigmoid(hidden_activations)
        hidden_states = (hidden_probs > np.random.rand(self.num_hidden)).astype(int)
        return hidden_states

    def contrastive_divergence(self, visible_data):
        # Positive phase
        positive_hidden = self.sample_hidden(visible_data)
        positive_associations = np.dot(visible_data.T, positive_hidden)

        # Negative phase
        negative_visible = self.sample_visible(positive_hidden)
        negative_hidden = self.sample_hidden(negative_visible)
        negative_associations = np.dot(negative_visible.T, negative_hidden)

        # Update weights and biases
        self.weights += self.learning_rate * (positive_associations - negative_associations) / self.batch_size
        self.visible_bias += self.learning_rate * np.sum(visible_data - negative_visible, axis=0) / self.batch_size
        self.hidden_bias += self.learning_rate * np.sum(positive_hidden - negative_hidden, axis=0) / self.batch_size

    def train(self, data, epochs):
        num_batches = len(data) // self.batch_size
        for epoch in range(epochs):
            np.random.shuffle(data)
            for batch in range(num_batches):
                batch_data = data[batch * self.batch_size:(batch + 1) * self.batch_size]
                self.contrastive_divergence(batch_data)

# Handwritten digit recognition
if __name__ == "__main__":

    # Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Boltzmann Machine
    num_visible = X_train.shape[1]
    num_hidden = 100
    bm = BoltzmannMachine(num_visible, num_hidden, learning_rate=0.01, batch_size=10)
    bm.train(X_train, epochs=10)

    # Use the trained Boltzmann Machine for classification
    y_pred = []
    for x in X_test:
        visible_states = x.reshape(1, -1)
        hidden_states = bm.sample_hidden(visible_states)
        reconstructed_visible = bm.sample_visible(hidden_states)
        y_pred.append(np.argmax(reconstructed_visible))

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")