import numpy as np
import matplotlib.pyplot as plt

def step_activation(x):
    return 1 if x >= 0 else 0

def perceptron(X, y, weights, learning_rate, max_epochs, tolerance):
    for epoch in range(max_epochs):
        sse = 0
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights[1:]) + weights[0]
            output = step_activation(net_input)
            error = target - output
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error
            sse += error**2
        if sse <= tolerance:
            return epoch + 1  # number of epochs taken
    return max_epochs  # if the function does not converged within limit

# Training data (AND function)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Parameters
initial_weights = np.array([10, 0.2, -0.75])
learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
max_epochs = 1000
tolerance = 0.002

epochs_taken = []

for lr in learning_rates:
    weights_copy = initial_weights.copy()  # reset weights 
    epochs = perceptron(X, y, weights_copy, lr, max_epochs, tolerance)
    epochs_taken.append(epochs)
    print(f"Learning rate={lr}, Epochs taken={epochs}")

# Plot learning rate vs. number of epochs
plt.plot(learning_rates, epochs_taken, marker="o")
plt.xlabel("Learning Rate")
plt.ylabel("Number of Epochs to Converge")
plt.title("Effect of Learning Rate on Convergence")
plt.show()
