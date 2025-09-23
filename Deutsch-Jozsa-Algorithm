import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

n = 3  # number of qubits

def dj_function(num_qubits):
 
    qc_dj = QuantumCircuit(num_qubits + 1)
    if np.random.randint(0, 2):
        # Flip output qubit with 50% chance
        qc_dj.x(num_qubits)
    if np.random.randint(0, 2):
        # Return constant circuit with 50% chance
        return qc_dj
 
    # If the "if" statement above was TRUE then we've returned the constant
    # function and the function is complete. If not, we proceed in creating our
    # balanced function:
 
    # Select half of all possible states at random
    on_states = np.random.choice(
        range(2**num_qubits),  # numbers to sample from
        2**num_qubits // 2,    # number of samples
        replace=False,         # ensures states are only sampled once
    )
 
    def add_cx(qc_dj, bit_string):
        for qubit, bit in enumerate(reversed(bit_string)):
            if bit == "1":
                qc_dj.x(qubit)
        return qc_dj
 
    for state in on_states:
        # Barriers are added to help visualize how the functions are created. They can safely be removed.
        qc_dj = add_cx(qc_dj, f"{state:0b}")
        qc_dj.mcx(list(range(num_qubits)), num_qubits)
        qc_dj = add_cx(qc_dj, f"{state:0b}")
 
    # qc_dj.barrier()
 
    return qc_dj

# 1. Create circuit
circuit = QuantumCircuit(n + 1, n)           # (number of qubits, number of classical bits)

# 2. Set output qubit n to 1
circuit.x(n)

# 3. Apply Hadamard gate to all qubits
circuit.h(range(n+1))

# 4. Call oracle
circuit.barrier()                            # visualize a barrier
oracle = dj_function(n)                      # call function & save in oracle
blackbox = oracle.to_gate()                  # converts the circuit into a "gate" so it can be inserted
circuit.compose(blackbox, inplace=True)      # oracle function is inserted into the circuit
circuit.barrier()                            # visualize a barrier

# 5. Apply second Hadamard gate, but only on inputs
circuit.h(range(n))

# 6. Measure the input qubits
circuit.measure(range(n), range(n))          # measure all input qubits and save to the classical bits

# 7. Simulation and output
# Backend + transpilation
backend = FakeAlmadenV2()                                                       # simulator we are using
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)        # makes the circuit runnable for the simulator
isa_circuit = pm.run(circuit)                                                   # save optimized circuit in isa_circuit

# Use Sampler
sampler = Sampler(backend)                   # sampler returns probability distribution of measurement results
job = sampler.run([isa_circuit], shots=1)    # execute, once
result = job.result()                        # save result in result

counts    = result[0].data.c.get_counts()    # e.g. {'111': 1}
print(counts)
bitstring = next(iter(counts))               # iter() creates list of all keys and next() gets first entry (only makes sense for one measurement)

print(f"Measured bitstring:")
if bitstring == '0'*n:
    print("→ Function was constant.")
else:
    print("→ Function was balanced.")
print("Circuit:")
display(circuit.draw("mpl"))
print("Oracle Circuit:")
display(oracle.draw("mpl"))
