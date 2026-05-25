import numpy as np
import matplotlib.pyplot as plt

def step_activation(x):
    return 1 if x >= 0 else 0

def perceptron(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    for epoch in range(max_epochs):
        sse = 0
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights[1:]) + weights[0]
            output = step_activation(net_input)
            error = target - output
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error
            sse += error**2
        errors.append(sse)
        if sse <= tolerance:
            print("Converged at epoch",epoch+1," with SSE=",sse)
            break
    return weights, errors

def predict(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = step_activation(net_input)
        outputs.append((xi, output))
    return outputs

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05
max_epochs = 1000
tolerance = 0.002

trained_weights, errors = perceptron(X, y, weights, learning_rate, max_epochs, tolerance)
predictions = predict(X, trained_weights)

for xi, output in predictions:
    print("Input:", xi, ", Output:", output)

plt.plot(range(1, len(errors)+1), errors, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence")
plt.show()
