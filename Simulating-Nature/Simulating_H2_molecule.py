# ==============================================================
# SECTION 1 — Problem Setup (Molecule, Basis Set, Active Space)
# ==============================================================
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper

# Define the H2 molecule with a bond length of 0.74 Å
# PySCFDriver performs the classical electronic structure calculation
R = 0.74  # bond length in Ångström
driver = PySCFDriver(atom=f"H 0 0 0; H 0 0 {R}", basis="STO-3G")

# Run the driver to obtain the second-quantized Hamiltonian and molecular data
problem = driver.run()

# The ActiveSpaceTransformer reduces the problem to a smaller active space
# Here we keep only 2 electrons in 2 spatial orbitals (H₂ minimal basis)
ast = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
problem = ast.transform(problem)

# Define how to map fermionic operators (electrons) to qubits.
# The ParityMapper is one of several options (others: Jordan–Wigner, Bravyi–Kitaev)
mapper = ParityMapper()


# ==============================================================
# SECTION 2 — Ansatz (Hartree–Fock + UCCSD)
# ==============================================================
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# Extract molecular parameters: number of electrons and orbitals
num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals

# Build the initial state based on the Hartree–Fock reference configuration
# This prepares the reference Slater determinant on qubits
initial = HartreeFock(num_spatial_orbitals, num_particles, mapper)

# Define the UCCSD ansatz (Unitary Coupled Cluster with Single and Double excitations)
# This parameterized quantum circuit will be optimized during VQE
ansatz = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=initial
)


# ==============================================================
# SECTION 3 — Circuit Optimization (Transpilation for Simulator)
# ==============================================================
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager

# Use the AerSimulator as backend (simulates an actual quantum device)
backend = AerSimulator()

# Generate a PassManager that optimizes the circuit for this backend
# Optimization level 3 performs aggressive gate reduction and layout selection
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

# Transpile (optimize) the ansatz circuit using the pass manager
ansatz = pm.run(ansatz)


# ==============================================================
# SECTION 4 — VQE Setup and Execution (on AerSimulator)
# ==============================================================
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, SPSA
from qiskit.primitives import BackendEstimatorV2
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import numpy as np

# Define number of shots to simulate measurement noise (realistic behavior)
shots = 8192

# Create an Estimator primitive for expectation value evaluations on the backend
estimator = BackendEstimatorV2(
    backend=backend,
)

# Choose the optimizer: SPSA (Simultaneous Perturbation Stochastic Approximation)
# SPSA is robust to noise and works well on shot-based simulations
optimizer = SPSA(maxiter=200)

# Define the VQE algorithm using the ansatz, optimizer, and estimator
vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=estimator)

# Combine the qubit mapper and VQE into a high-level Ground State Solver
solver = GroundStateEigensolver(mapper, vqe)

# Solve the electronic structure problem (estimate ground-state energy)
res = solver.solve(problem)


# ==============================================================
# SECTION 5 — Energy Results
# ==============================================================
# Extract results: electronic, nuclear, and total (electronic + nuclear repulsion) energies
E_elec = res.electronic_energies[0].real
E_nuc = res.nuclear_repulsion_energy
E_tot = res.total_energies[0].real

# Print the computed energies from the VQE simulation
print("\n=== VQE Result (on AerSimulator) ===")
print(f"Total energy (incl. nuclear repulsion): {E_tot:.8f} Hartree")
print(f"Nuclear repulsion: {E_nuc:.8f} Hartree")
print(f"Electronic energy: {E_elec:.8f} Hartree")

# Reference Full Configuration Interaction (FCI) values for comparison
print("\n=== FCI results ===")
print(f"Total energy (incl. nuclear repulsion): -1.13728 Hartree")
print(f"Nuclear repulsion:  0.71510 Hartree")
print(f"Electronic energy: -1.85238 Hartree")


# ==============================================================
# SECTION 6 — Statevector and Probabilities
# ==============================================================
from qiskit.quantum_info import Statevector

# Retrieve the optimized parameters from the VQE result
theta_opt = res.raw_result.optimal_point

# Reconstruct the final ground-state wavefunction using the optimized parameters
psi = Statevector.from_instruction(ansatz.assign_parameters(theta_opt))

# Print the entire quantum statevector (complex amplitudes)
print("\n--- Ground State Wavefunction (Full Statevector) ---")
print(psi)

# Extract the amplitude data and compute probabilities for each computational basis state
amps = psi.data
probs_percent = (np.abs(amps) ** 2) * 100.0

# Print formatted table: basis states, amplitudes, absolute values, and probabilities
print("\n--- Basis States and Probabilities (%) ---")
print(f"{'Basis state':>10} | {'Amplitude (Re,Im)':>20} | {'|Amplitude|':>12} | {'Probability (%)':>20}")
print("-" * 85)
for i in range(len(probs_percent)):
    bitstring = f"|{i:04b}>"  # Format index as a 4-bit binary string
    a_re = amps[i].real
    a_im = amps[i].imag
    abs_a = np.abs(amps[i])
    p = round(probs_percent[i], 2)
    print(f"{bitstring:>10}  | {a_re:+.6f}{a_im:+.6f}j  | {abs_a:12.6f} | {p:18.2f}")
print("-" * 85)
print(f"Sum of probabilities = {np.sum(probs_percent):.2f} %")


# ==============================================================
# SECTION 7 — Visualization
# ==============================================================
# Draw the ansatz circuit using Matplotlib visualization
# fold=-1 ensures that the entire circuit is drawn on one horizontal line
ansatz.draw("mpl", fold=-1)
