import math
def summationunit(inputs,weights,bias):
    sum1=0
    for (i,w) in zip(inputs,weights):
        sum1+=i*w
    return sum1+bias
def step_activation(x):
    return 1 if x>=0 else 0
def bipolarstep_activation(x):
    return 1 if x>=0 else -1
def tanh_activation(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)-math.exp(-x))
def sigmoid_activation(x):
    return 1/(1+math.exp(-x))
def relu_activation(x):
    return max(0,x)
def leakyrelu_activation(x):
    return max(0.1*x,x)

def comparator(target,prediction):
    error=target-prediction
    mse=(error^2)^0.5
    return error,mse

inp=[0.5,-0.5,0.6]
weights=[0.7,0.8,0.9]
bias=0.3
net_input=summationunit(inp,weights,bias)
print("aggregation function:",summationunit(inp,weights,bias))
print("step function:",step_activation(net_input))
print("bipolar step function:",bipolarstep_activation(net_input))
print("tanh function:",tanh_activation(net_input))
print("sigmoid function:",sigmoid_activation(net_input))
print("relu function:",relu_activation(net_input))
print("leakyrelu function:",leakyrelu_activation(net_input))
target=1
print("error for step activation function:",1-step_activation(net_input))
print("error for bipolar step activation function:",1-bipolarstep_activation(net_input))
print("error for tanh activation function:",1-tanh_activation(net_input))
print("error for sigmoid activation function:",1-sigmoid_activation(net_input))
print("error for relu activation function:",1-relu_activation(net_input))
print("error for leakyrelu activation function:",1-leakyrelu_activation(net_input))


