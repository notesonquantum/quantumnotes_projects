# Deutsch–Jozsa Algorithm (Qiskit 2.1.1)

This README provides a **short overview** of:

1. What the **Deutsch–Jozsa problem** is,  
2. What is explained in each of the **three accompanying videos**, and  
3. The **overall structure** and **execution flow** of the Qiskit code implementation.

The **actual Python implementation** of the algorithm can be found here:  
👉 [**deutsch_jozsa_algorithm_code.py**](deutsch_jozsa_algorithm_code.py)

---

## 🧠 Idea

We are given an **oracle** — a black-box function:

> f(x): {0,1}ⁿ → {0,1}

It takes a bit string of length **n** as input and outputs either **0** or **1**.

The oracle can represent **three possible functions**:

1. **Constant 0:** returns 0 for all possible inputs  
2. **Constant 1:** returns 1 for all possible inputs  
3. **Balanced:** returns 0 for exactly half of all inputs and 1 for the other half

Our goal is to determine **whether the function is constant or balanced**.

---

### ⚙️ Classical approach

Classically, this can only be solved by **testing multiple inputs** and checking their outputs.  
In the **worst case**, you need to test **half of all possible inputs plus one**, that is:

> 2^(n-1) + 1 evaluations of f(x)

Only then can you be certain whether the function is constant or balanced.  
This means the classical runtime **grows exponentially** with the input size.

🎬 [**Why Quantum Computing May Be So Powerful – Deutsch-Jozsa Algorithm Explained (Part 1)**](https://youtu.be/Dq0NfQWpe6k)  
*Explains the problem setup and the limits of the classical approach.*

---

### ⚛️ Quantum approach

With a **quantum computer**, we can find out whether the function is constant or balanced  
using **just a single query** to the oracle.

Key ideas:
- Quantum **superposition** allows all inputs to be evaluated simultaneously  
- **Interference** encodes whether the function is constant or balanced  
- After one oracle call and one measurement, the result is clear

🎬 [**How the Deutsch-Jozsa Algorithm Works (Part 2)**](https://youtu.be/eYD-fQEBx2U)   
*Shows how superposition and interference make the single-query solution possible.*

---

### 💻 Implementation

In the final part, we translate the algorithm into **Qiskit 2.1.1**,  
build the oracle, apply Hadamard gates, and run the circuit with **Sampler V2** on a fake backend.

🎬 [**Implementing the Deutsch-Jozsa Algorithm in Code (Part3)**](https://youtu.be/4kXjGTEufXg)  
*Demonstrates the complete Deutsch–Jozsa algorithm in Python.*

---

## ⚙️ Implementation Details

- **Language / Framework:** Python 3.10 +, Qiskit 2.1.1  
- **Backend:** `FakeAlmadenV2` (simulator)
- **Primitive:** `SamplerV2` (probabilistic execution)
- **Transpilation:** `generate_preset_pass_manager()` for ISA compatibility
- **Random Oracle:** The script builds either a *constant* or *balanced* oracle at random each time.

---

## ⚙️ Circuit Steps

1. **Create the quantum circuit**  
   Initialize a quantum circuit with \( n + 1 \) qubits (input + output) and \( n \) classical bits for measuring the input qubits.

2. **Set the output qubit to |1⟩**  
   Apply an **X-gate** to the last qubit (index \( n \)) so that the output qubit starts in the |1⟩ state.

3. **Create superposition**  
   Apply **Hadamard gates** to **all qubits** (input and output) to create an equal superposition over all input states.

4. **Call the oracle function**  
   - Insert a **barrier** for visual clarity.  
   - Generate a random oracle with `dj_function(n)` that is **constant** or **balanced** (50% probability).  
   - Convert the oracle circuit into a **gate** (`to_gate()`) and **compose** it into the main circuit (`compose()`).

5. **Interference step**  
   - Insert another **barrier** after the oracle.  
   - Apply **Hadamard gates** again, but **only on the input qubits**, to generate interference that encodes whether the oracle is constant or balanced.

6. **Measurement**  
   Measure all **input qubits** and store their outcomes in the corresponding **classical bits** (`measure(range(n), range(n))`).

7. **Simulation and evaluation**  
   - Use `FakeAlmadenV2()` as the simulator backend.  
   - Generate a **preset pass manager** with `generate_preset_pass_manager()` to transpile the circuit.  
   - Execute the transpiled circuit with the **SamplerV2** (`shots=1`).  
   - Retrieve the measurement results using `get_counts()`.  
     - If the bitstring is `000...0` → the function is **constant**.  
     - Otherwise → the function is **balanced**.
