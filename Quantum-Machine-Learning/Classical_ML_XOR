# MINI-PROJECT: LEARNING XOR
# Goal: write a small program that learns the XOR function
# 1 hidden layer, a few neurons
# no libraries, no numpy, no pytorch
# import packages we need (would also work without, but useful for 
# understanding neural networks)

import math # math lib for e.g. exp() function, not known to Python otherwise
import random # we need this for random numbers

# ----------------------------------
# Step 1: Define functions

# sigmoid function maps any number to the range 0–1
# sigma(x) = 1 / ( 1 + exp(-x) )
# nowadays ReLU is often used to avoid gradient vanishing

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# for backpropagation we need the gradient
# so we define the derivative of sigmoid

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# ----------------------------------
'''
# Step 2: Build network structure
# 2 inputs, 1 hidden layer with 2 neurons, 1 output (0 or 1)
# we need:
# weight matrix W1 to connect input with hidden layer
# bias vector b1 for hidden layer
# weight matrix W2 to connect hidden layer with output
# bias b2 for output
'''

# initialize weights randomly with small values
# random.uniform(-1,1) generates a random float between -1 and 1

# Input → Hidden Layer
W1 = [
    [random.uniform(-1, 1), random.uniform(-1, 1)],
    [random.uniform(-1, 1), random.uniform(-1, 1)]
]
B1 = [random.uniform(-1, 1), random.uniform(-1, 1)]

# Hidden Layer → Output
W2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
B2 = random.uniform(-1, 1)

# ----------------------------------
'''
# Step 3: Forward pass
# the goal of the forward pass is to send inputs through the network
# compute activations in the hidden layer and in the output layer
# and get the final output

# z1 is the weighted sum of the inputs for the first neuron
# we pass in an input x (an array with two entries, each 0 or 1,
# see XOR truth table)
# then compute: input1 * weight + input2 * weight + bias

# same for neuron 2
# finally, apply the same principle for the output neuron
'''

def forward(x):
    # Hidden Layer
    z1 = x[0] * W1[0][0] + x[1] * W1[1][0] + B1[0]
    a1 = sigmoid(z1)

    z2 = x[0] * W1[0][1] + x[1] * W1[1][1] + B1[1]
    a2 = sigmoid(z2)

    # Output Layer
    z3 = a1 * W2[0] + a2 * W2[1] + B2
    a3 = sigmoid(z3)

    return z1, a1, z2, a2, z3, a3

'''    
# if we run this now, we only get random numbers → the network is not trained yet
# therefore we want to know how far each output is from the correct result
'''

# ----------------------------------
'''
# Step 4: Loss function
# we compute the error
# note: output will lie between 0 and 1, but should be 0 OR 1
# so the difference will be negative if the expected result is 1
# therefore we square it, so the error is always positive
'''

def loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

# ----------------------------------
'''
# Step 5: Backpropagation
# now we must adjust the weights so that this error becomes smaller
# if we repeat this often enough, the error becomes so small
# that in e.g. 99% of cases we get the correct result
# for this we compute the derivative of the loss function with respect
# to all weights and biases
# this yields a vector, where each entry is the corresponding derivative
# this vector points in the direction where the loss function increases most
# → this is called the gradient
# so we go in the opposite direction (since we want to minimize the error)
'''

def backward(x, y_true):
    global W1, B1, W2, B2

    z1, a1, z2, a2, z3, a3 = forward(x)  # single forward pass with all intermediate values

    # compute all derivatives
    dL_da3 = -2 * (y_true - a3)
    da3_dz3 = sigmoid_derivative(z3)

    dL_dW2 = [
        dL_da3 * da3_dz3 * a1,
        dL_da3 * da3_dz3 * a2
    ]
    dL_dB2 = dL_da3 * da3_dz3

    dL_dz1 = dL_da3 * da3_dz3 * W2[0] * sigmoid_derivative(z1)
    dL_dz2 = dL_da3 * da3_dz3 * W2[1] * sigmoid_derivative(z2)

    dL_dW1 = [
        [dL_dz1 * x[0], dL_dz2 * x[0]],
        [dL_dz1 * x[1], dL_dz2 * x[1]]
    ]
    dL_dB1 = [dL_dz1, dL_dz2]

    for i in range(2):
        for j in range(2):
            W1[i][j] -= learning_rate * dL_dW1[i][j]
    for i in range(2):
        B1[i] -= learning_rate * dL_dB1[i]
    for i in range(2):
        W2[i] -= learning_rate * dL_dW2[i]
    B2 -= learning_rate * dL_dB2

# ----------------------------------
'''
# Step 6: Training data + parameters
# we need to know how far we go in this direction per step → learning rate
# it should not be too large (otherwise it won’t converge) but also not too small (too slow)
# there are special functions for this, here we just choose 0.1
# we also need the number of iterations, here we choose 10,000
'''

learning_rate = 0.1  # step size when adjusting weights
epochs = 10000       # number of iterations

# now define the training data: for input (x,y) the output should be z

# training data
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# ----------------------------------
# Step 7: Training – now the actual training

for epoch in range(epochs):
    for x, y in data:
        backward(x, y)

    # optional: show error every 1000 epochs
    if epoch %  1000 == 0:
        total_loss = 0
        for x, y in data:
            y_pred = forward(x)[-1]  # a3 is the last element
            total_loss += (y - y_pred) ** 2
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

# ----------------------------------
# Step 8: Testing after training with repetitions

runs = 1000  # how many times each input is tested

print("\nTest after training:")
for x, y in data:
    correct = 0
    for _ in range(runs):
        y_pred = forward(x)[-1]  # only a3 matters here
        y_class = 1 if y_pred >= 0.5 else 0  # or 0.49 if you want some tolerance
        if y_class == y:
            correct += 1
    print(f"Input: {x} → Expected: {y} | Correct: {correct}/{runs} ({correct/runs*100:.1f}%)")
    print(f"Input: {x}, y_pred: {y_pred:.10f}, rounded: {y_class}, expected: {y}")
