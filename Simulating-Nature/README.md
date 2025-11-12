# ⚛️ Simulating Nature – Variational Quantum Eigensolver (Qiskit 2.1.1)

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

---

### ⚙️ Classical approach

Classically, finding molecular ground-state energies requires solving the **Schrödinger equation**  
for interacting electrons — a task that scales **exponentially** with the system size.  
Methods such as **Full Configuration Interaction (FCI)** are exact but computationally intractable  
for all but the smallest molecules.

---

### ⚛️ Quantum approach

The **VQE** provides a quantum solution to this problem.  
Instead of representing the entire wavefunction on a classical computer,  
it uses a **parameterized quantum circuit (ansatz)** to approximate the ground state.  
A **quantum backend** evaluates the energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩,  
and a **classical optimizer** updates the parameters θ to minimize it.

Key ideas:
- Use a **hybrid loop** between quantum and classical computation.  
- Exploit **superposition and entanglement** to efficiently explore the Hilbert space.  
- Find the **ground-state energy** as the global minimum of the variational cost function.

🎬 [**Simulating Nature – The Variational Quantum Eigensolver (VQE)**](https://www.youtube.com/@notesonquantum)  
*Explains how the VQE algorithm approximates molecular ground-state energies using Qiskit Nature.*

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
- **Framework:** Qiskit 2.1.1 + Qiskit Nature  

---

## ⚙️ Execution Flow

1. **Problem setup**  
   Define the H₂ molecule using `PySCFDriver` and reduce it to an active space (2e⁻, 2 orbitals).

2. **Mapping to qubits**  
   Use the `ParityMapper` to transform the fermionic Hamiltonian into a qubit Hamiltonian.

3. **Ansatz preparation**  
   Build the initial **Hartree–Fock state** and the **UCCSD ansatz** to parameterize the wavefunction.

4. **Circuit optimization**  
   Transpile and optimize the ansatz circuit with  
   `generate_preset_pass_manager(optimization_level=3)` for the `AerSimulator`.

5. **VQE setup**  
   Combine the ansatz, `BackendEstimatorV2`, and the `SPSA` optimizer to construct the hybrid VQE algorithm.

6. **Solve the ground-state problem**  
   Use `GroundStateEigensolver` to minimize the expectation value and estimate the molecule’s energy.

7. **Energy analysis**  
   Print the electronic, nuclear, and total energies and compare them with the exact FCI reference.

8. **Wavefunction reconstruction**  
   Retrieve the optimized parameters and reconstruct the final statevector to analyze  
   amplitudes, probabilities, and the superposition structure of the ground state.

9. **Visualization**  
   Plot the optimized ansatz circuit with Matplotlib (`fold=-1` for full horizontal view).

---

## ⚙️ Output Summary

- Prints **VQE energy results** and **FCI reference values**  
- Displays the **optimized statevector** and **probability distribution** over all basis states  
- Confirms the **sum of probabilities = 100%**  
- Visualizes the **quantum circuit** representing the final ansatz

---

📘 *This simulation shows how quantum algorithms can reproduce physical reality at the molecular level —  
demonstrating the power of Qiskit Nature to simulate nature itself.*

---
