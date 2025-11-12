# ⚛️ Simulating Nature – Variational Quantum Eigensolver (Qiskit 1.4.4)

This README provides a **short overview** of:

1. What the **Variational Quantum Eigensolver (VQE)** algorithm is,  
2. What is explained in the **corresponding video**, and  
3. The **overall structure** and **execution flow** of the Qiskit code implementation.

The **actual Python implementation** of the simulation can be found here:  
👉 [**simulating_nature_vqe.py**](simulating_nature_vqe.py)

---

## 🧠 Idea

The **Variational Quantum Eigensolver (VQE)** is a **hybrid quantum-classical algorithm** used to find  
the **ground-state energy** of a molecule — in this case, the **hydrogen molecule (H₂)**.  
It combines the power of a quantum computer to represent complex wavefunctions  
with a classical optimizer that adjusts the parameters to minimize the energy expectation value.

The main idea is to **simulate nature itself**, since quantum systems can naturally model other quantum systems.  
The algorithm finds the minimal energy configuration — the **ground state** — by variationally  
adjusting a parameterized quantum circuit until it best represents the molecule’s lowest-energy wavefunction.

🎬 [**Simulating Nature with Quantum Computers Part 1 - The Big Picture**](https://youtu.be/Aoi6DwUw9zQ)  

---

### ⚙️ Classical approach

Classically, finding molecular ground-state energies requires solving the **Schrödinger equation**  
for interacting electrons — a task that scales **exponentially** with the system size.  
Methods such as **Full Configuration Interaction (FCI)** are exact but computationally intractable  
for all but the smallest molecules.

🎬 [**Simulating Nature with Quantum Computers Part 2 - The Technical Workflow**](https://youtu.be/YAdT5Z4Tmsw)  

---

### ⚛️ Quantum approach

The **VQE** provides a quantum solution to this problem.  
Instead of representing the entire wavefunction on a classical computer,  
it uses a **parameterized quantum circuit (ansatz)** to approximate the ground state.  
A **quantum backend** evaluates the energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩,  
and a **classical optimizer** updates the parameters θ to minimize it.

🎬 [**Simulating Nature with Quantum Computers Part 3 - The Variational Quantum Eigensolver**](https://youtu.be/dejBlXnZNHM)  

---

## ⚙️ Implementation Details

- **Molecule:** Hydrogen (H₂) with bond length **0.74 Å**  
- **Basis set:** STO-3G (minimal basis)  
- **Active space:** 2 electrons in 2 orbitals  
- **Mapper:** ParityMapper (fermion-to-qubit transformation)  
- **Ansatz:** Hartree–Fock reference state + UCCSD (Unitary Coupled Cluster Singles and Doubles)  
- **Optimizer:** SPSA (Simultaneous Perturbation Stochastic Approximation)  
- **Primitive:** `BackendEstimatorV2`  
- **Backend:** `AerSimulator` (shot-based simulation, 8192 shots)  
- **Framework:** Qiskit 1.4.4 + Qiskit Nature  

---

## ⚙️ Execution Flow

🎬 [**Simulating Nature with Quantum Computers Part 4 — How to actually simulate it**](https://youtu.be/kwFZLDZKYSw)  

1. **Problem setup**  
   Define the H₂ molecule (bond length 0.74 Å) using `PySCFDriver`,  
   reduce it to 2 electrons in 2 orbitals with `ActiveSpaceTransformer`,  
   and map it to qubits using the `ParityMapper`.

2. **Ansatz preparation**  
   Build the **Hartree–Fock** reference state and define the **UCCSD ansatz**  
   (Unitary Coupled Cluster Singles and Doubles) as the parameterized circuit.

3. **Circuit optimization**  
   Use the `AerSimulator` as backend and transpile the ansatz with  
   `generate_preset_pass_manager(optimization_level=3)` for gate optimization.

4. **VQE setup and execution**  
   Combine the ansatz, `BackendEstimatorV2`, and the **SPSA optimizer**  
   within the **VQE algorithm**, and solve the ground-state energy with  
   `GroundStateEigensolver(mapper, vqe)`.

5. **Energy results**  
   Print the total, nuclear, and electronic energies and compare them  
   to reference FCI values for verification.

6. **Wavefunction reconstruction**  
   Retrieve the optimized parameters, reconstruct the statevector,  
   and display amplitudes and probabilities for each basis state.

7. **Visualization**  
   Draw the optimized ansatz circuit using Matplotlib (`fold=-1`).

---

## ⚙️ Output Summary

- Prints the **total energy** (including nuclear repulsion), the **nuclear repulsion energy**, and the **electronic energy** obtained from the VQE simulation  
- Displays the **reference FCI values** for direct comparison with the simulated results  
- Lists the **amplitudes and probabilities** of all computational basis states in the optimized ground state  
- Confirms that the **sum of probabilities equals 100 %**  
- Visualizes the **optimized ansatz circuit** representing the final ground-state wavefunction

---
