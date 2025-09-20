import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Use either of these as target
y_and = np.array([[1,0],[1,0],[1,0],[0,1]])
y_xor = np.array([[1,0],[0,1],[0,1],[1,0]])

y = y_and # For AND gate; for XOR use y = y_xor

alpha = 0.05
max_epochs = 2000
tolerance = 0.002

np.random.seed(42)
v = np.random.uniform(-1,1,(2,2))
w = np.random.uniform(-1,1,(2,2))

errors = []
for epoch in range(max_epochs):
    sse = 0
    for idx in range(len(X)):
        A,B = X[idx]
        target = y[idx]
        net_h1 = v[0,0]*A + v[1,0]*B
        net_h2 = v[0,1]*A + v[1,1]*B
        h1 = sigmoid(net_h1)
        h2 = sigmoid(net_h2)
        net_o1 = w[0,0]*h1 + w[1,0]*h2
        net_o2 = w[0,1]*h1 + w[1,1]*h2
        o1 = sigmoid(net_o1)
        o2 = sigmoid(net_o2)
        error1 = target[0] - o1
        error2 = target[1] - o2
        sse += error1**2 + error2**2
        delta_o1 = error1 * sigmoid_derivative(net_o1)
        delta_o2 = error2 * sigmoid_derivative(net_o2)
        delta_h1 = (delta_o1 * w[0,0] + delta_o2 * w[0,1]) * sigmoid_derivative(net_h1)
        delta_h2 = (delta_o1 * w[1,0] + delta_o2 * w[1,1]) * sigmoid_derivative(net_h2)
        w[0,0] += alpha * delta_o1 * h1
        w[1,0] += alpha * delta_o1 * h2
        w[0,1] += alpha * delta_o2 * h1
        w[1,1] += alpha * delta_o2 * h2
        v[0,0] += alpha * delta_h1 * A
        v[1,0] += alpha * delta_h1 * B
        v[0,1] += alpha * delta_h2 * A
        v[1,1] += alpha * delta_h2 * B
    errors.append(sse)
    if sse <= tolerance:
        print("Converged at epoch", epoch+1, "with SSE=", sse)
        break
if sse > tolerance:
        print("Not Converging")

for idx in range(len(X)):
    A,B = X[idx]
    net_h1 = v[0,0]*A + v[1,0]*B
    net_h2 = v[0,1]*A + v[1,1]*B
    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)
    net_o1 = w[0,0]*h1 + w[1,0]*h2
    net_o2 = w[0,1]*h1 + w[1,1]*h2
    o1 = sigmoid(net_o1)
    o2 = sigmoid(net_o2)
    label = [int(o1 >= 0.5), int(o2 >= 0.5)]
    print("Input:", X[idx], "Output:", label)

plt.plot(range(1, len(errors)+1), errors, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error")
plt.title("Backpropagation NN for 2-output Logic Gate")
plt.show()
