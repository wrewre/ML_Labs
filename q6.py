import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

def perceptron(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    for epoch in range(max_epochs):
        sse = 0
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights[1:]) + weights[0]
            output = sigmoid_activation(net_input)
            error = target - output
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error
            sse += error**2
        errors.append(sse)
        if sse <= tolerance:
            print("Converged at epoch", epoch+1, "with SSE=", sse)
            break
    return weights, errors

def predict(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = sigmoid_activation(net_input)
        label = 1 if output >= 0.5 else 0
        outputs.append((xi, label))
    return outputs

# Customer dataset (Candies, Mangoes, Milk Packets, Payment)
X = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 4, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

# Labels (High Value? Yes=1, No=0)
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Initialize weights & parameters
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # One bias + 4 features
learning_rate = 0.001
max_epochs = 2000
tolerance = 0.01

trained_weights, errors = perceptron(X, y, weights, learning_rate, max_epochs, tolerance)
predictions = predict(X, trained_weights)

print("Predictions (Inputs, Output Label):")
for xi, output in predictions:
    print("Input:", xi, "Output:", "High Value" if output == 1 else "Low Value")

plt.plot(range(1, len(errors)+1), errors, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence (Sigmoid Activation)")
plt.show()
