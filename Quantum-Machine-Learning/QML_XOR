'''
The principle behind the code is as follows:
We generate a statevector that describes both qubits, which represent the inputs,
through entanglement. This vector is "shaped" so that it already stores a certain pattern,
namely equal and unequal.  
In this case, the statevector can only collapse at the end into 00 and 11 or
into 01 and 10.
'''

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from dataclasses import dataclass
import numpy as np

# Step 1: Define dataset - 4 vectors corresponding to 0 or 1

# 4 points (XOR)
X = np.array([
    [0, 0],  # -> -1
    [0, 1],  # -> +1
    [1, 0],  # -> +1
    [1, 1],  # -> -1
])

# Labels (target outputs)
y = np.array([-1, 1, 1, -1])

print("X:\n", X)
print("y:", y)

# Step 2: Data encoding - data must be made quantum-friendly
# values are encoded into qubit states, 0 → state |0>, 1 → state |1>

def feature_map_circuit(x):
    qc = QuantumCircuit(2)
    for i in range(2):
        if x[i] == 1.0:
            qc.x(i)
    return qc
    
# Step 3: Ansatz with parameters
# in this step we create the first and second layer
# we take our 2 qubits and apply an operation depending on parameters 0,1,2,3,
# then we entangle them
# then comes the second layer

def variational_ansatz(theta):
    """
    theta: Array of length 4
      - theta[0]: RY on qubit 0 (first layer)
      - theta[1]: RY on qubit 1 (first layer)
      - CX(0->1): entanglement
      - theta[2]: RY on qubit 0 (second layer)
      - theta[3]: RY on qubit 1 (second layer)
    """
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    # redundant, because this only changes the relation between 00 and 01 or 10 and 11
    # and since we only measure qubit 0 this is unnecessary (in contrast, if we measured
    # only qubit 1 then theta[2] would be unnecessary)
    # qc.ry(theta[3], 1)
    return qc

# Step 4: Build model circuit
# simply combine everything so far

def build_model_circuit(x, theta):

    qc = QuantumCircuit(2)
    # Feature encoding (write data into qubits)
    qc.compose(feature_map_circuit(x), inplace=True)
    # Trainable ansatz
    qc.compose(variational_ansatz(theta), inplace=True)

    return qc

# Step 5: Prediction by measurement
# first we define the operator Z_q0, which measures qubit 0 in the Z basis and returns -1
# for state |1> and +1 for state |0>, then we compute the expectation value of the vector
# which lies somewhere between -1 and +1...
# training works like this: e.g. we input 00 or 11, our parameters that rotate the vector
# around the y-axis should tune the vector so that when we measure in the Z-basis
# we always get -1, i.e. the vector collapses to |1> upon measurement,
# and for 01 and 10 it should collapse to |0>,
# meaning the measurement then outputs +1

# in the function we then build the circuit, compute the statevector,
# then compute the expectation value which must lie between -1 and +1
# finally we return it; clip ensures values stay in [-1, +1] in case of rounding errors

# Observable: Z on qubit 0 in a 2-qubit system
Z_q0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=2)

def predict_expectation(x, theta):

    qc  = build_model_circuit(x, theta)
    psi = Statevector.from_instruction(qc)
    exp = np.real(psi.expectation_value(Z_q0))
    return float(np.clip(exp, -1.0, 1.0))

def predict_label(x, theta):
    z_exp = predict_expectation(x, theta)
    return 1 if z_exp >= 0 else 0

# Step 6: Loss & Training

# mse_loss computes the loss by creating an array where the
# expectation value is computed for each input; this vector is then
# subtracted by the labels, squared (to make all differences positive),
# and finally averaged so we get a single number
def mse_loss(X, y, theta):
    
    preds = np.array([predict_expectation(x, theta) for x in X])
    return float(np.mean((preds - y) ** 2))

# in numerical_grad we compute the gradient
# we take the parameter vector theta and for each entry compute plus epsilon and minus epsilon,
# then subtract and divide by 2*epsilon, yielding the symmetric difference quotient,
# which gives the corresponding entry of the gradient
def numerical_grad(X, y, theta, eps):
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        t_plus  = theta.copy(); t_plus[j]  += eps
        t_minus = theta.copy(); t_minus[j] -= eps
        f_plus  = mse_loss(X, y, t_plus)
        f_minus = mse_loss(X, y, t_minus)
        grad[j] = (f_plus - f_minus) / (2*eps)
    return grad

@dataclass
class TrainConfig:
    lr: float = 0.25   # learning rate
    steps: int = 100   # number of iterations
    eps: float = 1e-3  # step size for gradients
    verbose: bool = True

def train_vqc(X, y, theta0, cfg: TrainConfig):
    theta = theta0.copy()
    history = []

    for t in range(cfg.steps):
        # compute loss
        loss = mse_loss(X, y, theta)
        history.append(loss)

        # print progress every 10% of steps
        if cfg.verbose and (t % (cfg.steps // 10 or 1) == 0 or t == cfg.steps - 1):
            print(f"Step {t:3d} | Loss = {loss:.4f} | Theta = {np.round(theta, 3)}")

        # gradient & update
        grad = numerical_grad(X, y, theta, eps=cfg.eps)
        theta -= cfg.lr * grad

    return theta, history

# Random initial values for 4 parameters (between -0.5 and 0.5)
theta_init = np.random.uniform(-0.5, 0.5, size=4)

# Training settings
cfg = TrainConfig(lr=0.25, steps=100, eps=1e-3, verbose=True)

# Start training:
# - train_vqc runs the training loop
# - returns the learned parameters (theta_star)
# - and the loss values per iteration (history)
theta_star, history = train_vqc(X, y, theta_init, cfg)

# Step 7: Results

# Predictions for all points
pred_exp = np.array([predict_expectation(x, theta_star) for x in X])
pred_lbl = np.array([predict_label(x, theta_star)      for x in X])

# Print pointwise results
print("Predictions:")
for i, x in enumerate(X):
    print(f"x={x} -> ⟨Z⟩={pred_exp[i]:+6.3f} | pred={pred_lbl[i]} | true={int((y[i]+1)/2)}")
    
# Compute accuracy
y_classical = ((y + 1) / 2).astype(int)
acc = np.mean(pred_lbl == y_classical)
print(f"\nAccuracy: {acc*100:.1f}%")

# (optional) learning curve check
print("Final loss:", mse_loss(X, y, theta_star))
