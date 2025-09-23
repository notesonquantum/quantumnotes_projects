import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from dataclasses import dataclass
from itertools import product
import matplotlib.pyplot as plt

np.random.seed(42)

# ----------------------
# 1. Datasets
# ----------------------
def generate_all_inputs(n_bits):
    return np.array(list(product([0.0, 1.0], repeat=n_bits)), dtype=float)

def parity_label(x):
    return int(sum(x) % 2)

def generate_dataset(n_bits):
    X = generate_all_inputs(n_bits)
    y = np.array([parity_label(x) for x in X])
    y_scaled = 2 * y - 1  # {-1, +1}
    return X, y_scaled

# ----------------------
# 2. Feature Map
# ----------------------
def feature_map_circuit(x):
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        if x[i] == 1.0:
            qc.x(i)
    return qc

# ----------------------
# 3. Variational Ansatz
# ----------------------
def variational_ansatz_shared(theta, n_qubits, depth):
    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        for i in range(n_qubits):
            qc.ry(theta[d], i)
        for i in reversed(range(n_qubits-1)):
            qc.cx(i+1, i)  
    return qc

# ----------------------
# 4. Combined Circuit
# ----------------------
def build_model_circuit_general(x, theta, depth):
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map_circuit(x), inplace=True)
    qc.compose(variational_ansatz_shared(theta, n_qubits, depth), inplace=True)
    return qc

def get_z_operator(n_qubits):
    return SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])

# ----------------------
# 5. Prediction
# ----------------------
def predict_expectation(x, theta, depth):
    n_qubits = len(x)
    qc = build_model_circuit_general(x, theta, depth)
    psi = Statevector.from_instruction(qc)
    Z_q0 = get_z_operator(n_qubits)
    exp = np.real(psi.expectation_value(Z_q0))
    return float(np.clip(exp, -1.0, 1.0))

def predict_label(x, theta, depth):
    return 1 if predict_expectation(x, theta, depth) >= 0 else 0

# ----------------------
# 6. Loss + Training
# ----------------------
def mse_loss(X, y, theta, depth):
    preds = np.array([predict_expectation(x, theta, depth) for x in X])
    return float(np.mean((preds - y) ** 2))

def numerical_grad(X, y, theta, depth, eps):
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        t_plus  = theta.copy(); t_plus[j]  += eps
        t_minus = theta.copy(); t_minus[j] -= eps
        f_plus  = mse_loss(X, y, t_plus, depth)
        f_minus = mse_loss(X, y, t_minus, depth)
        grad[j] = (f_plus - f_minus) / (2 * eps)
    return grad

@dataclass
class TrainConfig:
    lr: float = 0.25
    steps: int = 1000
    eps: float = 1e-3
    verbose: bool = True

def train_vqc(X, y, theta0, depth, cfg: TrainConfig):
    theta = theta0.copy()
    history = []

    for t in range(cfg.steps):
        loss = mse_loss(X, y, theta, depth)
        history.append(loss)

        if cfg.verbose and (t % (cfg.steps // 10 or 1) == 0 or t == cfg.steps - 1):
            print(f"Step {t:3d} | Loss = {loss:.4f} | Theta = {np.round(theta, 3)}")

        grad = numerical_grad(X, y, theta, depth, eps=cfg.eps)
        theta -= cfg.lr * grad

    return theta, history

# ----------------------
# 7. Training (n = 3)
# ----------------------
n_train_bits = 3
depth = 10

X_train, y_train = generate_dataset(n_train_bits)
theta_init = np.random.uniform(-0.5, 0.5, size=depth)
cfg = TrainConfig()

theta_star, history = train_vqc(X_train, y_train, theta_init, depth, cfg)

# ----------------------
# 8. Training Evaluation
# ----------------------
pred_exp = np.array([predict_expectation(x, theta_star, depth) for x in X_train])
pred_lbl = np.array([predict_label(x, theta_star, depth)      for x in X_train])
y_classical = ((y_train + 1) / 2).astype(int)

print("\nPredictions:")
for i, x in enumerate(X_train):
    print(f"x={x.astype(int)} -> ⟨Z⟩={pred_exp[i]:+6.3f} | pred={pred_lbl[i]} | true={y_classical[i]}")

acc = np.mean(pred_lbl == y_classical)
print(f"\nTrain Accuracy: {acc*100:.1f}%")
print("Final Loss:", mse_loss(X_train, y_train, theta_star, depth))

plt.plot(history)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss (Shared Ansatz)")
plt.grid(True)
plt.show()

# ----------------------
# Generalization test
# ----------------------

min_n = 2
max_n = 9

print("\nGeneralization test for n =", list(range(min_n, max_n + 1)))

results = []

for n in range(min_n, max_n + 1):
    X_test = generate_all_inputs(n)
    y_test = np.array([parity_label(x) for x in X_test])

    y_exp  = np.array([predict_expectation(x, theta_star, depth) for x in X_test])
    y_pred = np.array([1 if z >= 0 else 0 for z in y_exp])

    acc = np.mean(y_pred == y_test)
    results.append((n, acc * 100))
    print(f"n={n:2d} | Num Tests={len(X_test):3d} | Accuracy: {acc * 100:.1f}%")
