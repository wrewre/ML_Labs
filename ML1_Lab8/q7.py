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
print("X size:", X.size)
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
X_pinv = np.linalg.pinv(X_bias)
print("Pseudoinverse X size:", X_pinv.size)
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
learning_rate = 0.001
max_epochs = 2000
tolerance = 0.01
trained_weights1, errors1 = perceptron(X, y, weights.copy(), learning_rate, max_epochs, tolerance)
predictions1 = predict(X, trained_weights1)
weights_pinv = np.dot(X_pinv, y)
def predict_with_weights(X, weights):
    outputs = []
    for xi in X:
        net_input = weights[0] + np.dot(xi, weights[1:])
        output = sigmoid_activation(net_input)
        label = 1 if output >= 0.5 else 0
        outputs.append((xi, label))
    return outputs
predictions2 = predict_with_weights(X, weights_pinv)
print("Predictions from iterative perceptron learning:")
for xi, output in predictions1:
    print("Input:", xi, "Output:", "High Value" if output == 1 else "Low Value")
print("Predictions from pseudoinverse weights:")
for xi, output in predictions2:
    print("Input:", xi, "Output:", "High Value" if output == 1 else "Low Value")
plt.plot(range(1, len(errors1)+1), errors1, marker="o", label="Perceptron Learning (SSE)")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence vs Pseudoinverse")
plt.legend()
plt.show()
