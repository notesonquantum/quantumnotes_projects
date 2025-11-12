# Deutsch–Jozsa Algorithm (Qiskit 2.1.1)

> Educational implementation as part of the **Quantum Notes – Qiskit QML Project** (Portfolio for ETH Zürich / TU Delft / TUM applications).

<p align="center">
  <a href="https://www.youtube.com/@notesonquantum">
    <img alt="YouTube – Quantum Notes" src="https://img.shields.io/badge/YouTube-Quantum%20Notes-red?logo=youtube"/>
  </a>
  <a href="https://qiskit.org/">
    <img alt="Qiskit" src="https://img.shields.io/badge/Qiskit-2.1.1-6929C4?logo=qiskit"/>
  </a>
</p>

---

## 🧩 Idea

The **Deutsch–Jozsa algorithm** demonstrates the exponential advantage of quantum algorithms for a simple decision problem:

> Given a Boolean function \( f(x) \) that is either *constant* or *balanced*, determine which one it is — using only **one oracle query**.

In the classical case, \( 2^{n-1} + 1 \) evaluations are required in the worst case.  
Quantumly, only **one** evaluation of the oracle is needed.

---

## ⚙️ Implementation Details

- **Language / Framework:** Python 3.10 +, Qiskit 2.1.1  
- **Backend:** `FakeAlmadenV2` (simulator)
- **Primitive:** `SamplerV2` (probabilistic execution)
- **Transpilation:** `generate_preset_pass_manager()` for ISA compatibility
- **Random Oracle:** The script builds either a *constant* or *balanced* oracle at random each time.

### Circuit Steps

1. Initialize \( n + 1 \) qubits (last one = output).  
2. Apply **X** to output qubit and **Hadamard** to all qubits.  
3. Insert the **oracle** (balanced or constant).  
4. Apply Hadamard again on the input qubits.  
5. Measure the input register.  
6. Interpret result:
   - `000...0` → constant  
   - anything else → balanced

---

## ▶️ How to Run

```bash
python deutsch_jozsa.py
