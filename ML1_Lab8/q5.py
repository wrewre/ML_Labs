import numpy as np
import matplotlib.pyplot as plt
# single perceptron with step activation question 2
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
y = np.array([0,1,1,0])
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

# trying with all the activation functions question 3

import numpy as np
import matplotlib.pyplot as plt
import math

def step_activation(x):
    return 1 if x >= 0 else 0

def bipolarstep_activation(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

def relu_activation(x):
    return max(0, x)


def perceptron_bipolar_step(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    converged_epoch = None
    for epoch in range(max_epochs):
        sse = 0
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights[1:]) + weights[0]
            output = bipolarstep_activation(net_input)
            error = target - output
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error
            sse += error**2
        errors.append(sse)
        if sse <= tolerance:
            converged_epoch = epoch + 1
            print("bipolar step activation function:")
            print("Converged at epoch", converged_epoch, "with SSE=", sse)
            break
    if converged_epoch is None:
        print("bipolar step activation function: Did not converge within", max_epochs, "epochs")
    return weights, errors


def perceptron_step(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    converged_epoch = None
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
            converged_epoch = epoch + 1
            print("step activation function:")
            print("Converged at epoch", converged_epoch, "with SSE=", sse)
            break
    if converged_epoch is None:
        print("step activation function: Did not converge within", max_epochs, "epochs")
    return weights, errors


def perceptron_sigmoid(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    converged_epoch = None
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
            converged_epoch = epoch + 1
            print("sigmoid activation function:")
            print("Converged at epoch", converged_epoch, "with SSE=", sse)
            break
    if converged_epoch is None:
        print("sigmoid activation function: Did not converge within", max_epochs, "epochs")
    return weights, errors


def perceptron_relu(X, y, weights, learning_rate, max_epochs, tolerance):
    errors = []
    converged_epoch = None
    for epoch in range(max_epochs):
        sse = 0
        for xi, target in zip(X, y):
            net_input = np.dot(xi, weights[1:]) + weights[0]
            output = relu_activation(net_input)
            error = target - output
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error
            sse += error**2
        errors.append(sse)
        if sse <= tolerance:
            converged_epoch = epoch + 1
            print("relu activation function:")
            print("Converged at epoch", converged_epoch, "with SSE=", sse)
            break
    if converged_epoch is None:
        print("relu activation function: Did not converge within", max_epochs, "epochs")
    return weights, errors


def predict_step(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = step_activation(net_input)
        outputs.append((xi, output))
    return outputs


def predict_bipolar_step(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = bipolarstep_activation(net_input)
        outputs.append((xi, output))
    return outputs


def predict_sigmoid(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = sigmoid_activation(net_input)
        outputs.append((xi, output))
    return outputs


def predict_relu(X, weights):
    outputs = []
    for xi in X:
        net_input = np.dot(xi, weights[1:]) + weights[0]
        output = relu_activation(net_input)
        outputs.append((xi, output))
    return outputs

trained_weights1, errors1 = perceptron_step(X, y, weights.copy(), learning_rate, max_epochs, tolerance)
predictions1 = predict_step(X, trained_weights1)


trained_weights2, errors2 = perceptron_sigmoid(X, y, weights.copy(), learning_rate, max_epochs, tolerance)
predictions2 = predict_sigmoid(X, trained_weights2)


trained_weights3, errors3 = perceptron_relu(X, y, weights.copy(), learning_rate, max_epochs, tolerance)
predictions3 = predict_relu(X, trained_weights3)


trained_weights4, errors4 = perceptron_bipolar_step(X, y, weights.copy(), learning_rate, max_epochs, tolerance)
predictions4 = predict_bipolar_step(X, trained_weights4)


print("step function")
for xi, output in predictions1:
    print("Input:", xi, ", Output:", output)

print("sigmoid function")
for xi, output in predictions2:
    print("Input:", xi, ", Output:", output)

print("relu function")
for xi, output in predictions3:
    print("Input:", xi, ", Output:", output)

print("bipolar function")
for xi, output in predictions4:
    print("Input:", xi, ", Output:", output)

plt.plot(range(1, len(errors1) + 1), errors1, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence (Step Activation)")
plt.show()

plt.plot(range(1, len(errors2) + 1), errors2, marker="s")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence (Sigmoid Activation)")
plt.show()

plt.plot(range(1, len(errors3) + 1), errors3, marker="+")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence (ReLU Activation)")
plt.show()

plt.plot(range(1, len(errors4) + 1), errors4, marker="+")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error (SSE)")
plt.title("Perceptron Training Convergence (Bipolar Step Activation)")
plt.show()
