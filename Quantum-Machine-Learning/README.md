# 🧠 Quantum Machine Learning Series

This README provides a **short overview** of the files contained in this folder.  
It lists the corresponding Python scripts that are part of the Quantum Machine Learning series  
and links to each of them for quick access.  
The README serves as a **complement to the YouTube videos**,  
where the theory and implementation of each part are explained in detail.

1️⃣ Classical Machine Learning (XOR)  
2️⃣ Quantum Machine Learning – Theory  
3️⃣ Quantum Machine Learning – Implementation (Qiskit)  
4️⃣ Scaling Up & Real Device Execution

---

## 🎯 Goal of the Series

The main goal of this series is to **demonstrate how Quantum Machine Learning works** —  
starting from the foundations of classical machine learning and gradually transitioning into the quantum domain.

The series aims to:

- Explain how **classical machine learning** operates, using the XOR problem as an example.  
- Show how a **classical problem or model** can be **translated** into a **quantum formulation** that can be solved on a quantum computer.  
- Introduce the **Variational Quantum Classifier (VQC)** as a core concept for quantum learning.  
- Demonstrate how the XOR problem can first be solved with **two input bits**, and then **scaled up** to larger input sizes (three to nine bits).  
- Finally, show how the complete setup can be executed on a **real IBM Quantum backend** using **Qiskit 2.1.1**.

In short, the purpose of this series is to **bridge the classical and quantum worlds** of machine learning —  
both conceptually and practically, through simulation, mathematical analysis, and real-device execution.

---

## 🧩 Structure of the Video Series

🎬 [**Quantum Machine Learning Part 1 - Basics of Neural Networks**](https://youtu.be/Z7M8XsJcqRs)  

In the first video, we build the **classical foundation**:

- Explain what the **XOR problem** is
- Build a **small feed-forward neural network** (two inputs, one output) in Python  
- Explain how **backpropagation**, **weights**, and **gradients** work  
- Train the network to correctly classify the XOR pattern

🧠 *This serves as the conceptual baseline before moving into the quantum world.*

👉 [`classical_ML_XOR.py`](./classical_ML_XOR.py)  

---

🎬 [**Quantum Machine Learning Part 2 - Understanding the Variational Quantum Classifier**](https://youtu.be/o_IAMNLOaZs)  

In the second video, we move to the **quantum theoretical side**:

- Introduce the **Variational Quantum Classifier (VQC)** as the quantum analogue of a neural network  
- Explain how **parameterized quantum circuits** can learn nonlinear decision boundaries  
- Describe how **classical optimizers** adjust the circuit parameters  
- Show how a **quantum model** can learn the XOR pattern through interference and superposition

⚛️ *This video focuses purely on the **conceptual understanding** – no code yet.*

👉 [`QML_XOR.py`](./QML_XOR.py)  

---

🎬 [**Quantum Machine Learning Part 3 - Implementing the Variational Quantum Classifier**](https://youtu.be/1LGkmxKEZM8)  

Here we **translate theory into code**:

- Implement the **VQC** in **Qiskit 2.1.1**  
- Use the **EstimatorV2** or **SamplerV2** primitive to measure expectation values  
- Train the quantum circuit to reproduce the XOR mapping  
- Visualize the training loss and classification results  

📘 *This part connects the mathematical model from Part 2 with an actual Qiskit implementation.*

👉 [`QML_XOR.py`](./QML_XOR.py)

---

🎬 [**Quantum Machine Learning Part 4 - Scaling Up the Variational Quantum Classifier**](https://youtu.be/64rk-371fLk)  

Finally, we **scale up** the quantum classifier:

- Increase the number of input bits from 2 to up to 9  
- Keep the same circuit architecture but with more qubits and parameters  
- Train on a subset of all possible inputs and **test generalization**  
- Run simulations on a **fake backend** and on a **real IBM Quantum device**

📈 *This demonstrates how scaling and noise affect the performance of quantum models.*

👉 [`QML_XOR_scaled_calculated.py`](./QML_XOR_scaled_calculated.py)  
👉 [`QML_XOR_scaled_simulated.py`](./QML_XOR_scaled_simulated.py)  
👉 [`instance.txt`](./instance.txt)

---

## ⚙️ Code Overview

| Part | File | Description |
|------|------|--------------|
| 1️⃣ Classical XOR | `classical_xor.py` | Classical neural network learning XOR (Python simulation) |
| 2️⃣ QML Theory | `vqc_theory.py` | Illustrates the structure of the Variational Quantum Classifier (no code execution) |
| 3️⃣ QML Implementation | `vqc_implementation.py` | Qiskit implementation of the VQC learning XOR |
| 4️⃣ QML Scaled | `vqc_scaled.py` | Scaled-up VQC (3–9 input bits, optional IBM backend) |
| 🔐 IBM Quantum Instance | `instance.txt` | Contains your personal IBM Quantum credentials (not uploaded publicly) |

---

Next, we can go through each of the four code files one by one  
and add dedicated subsections just like we did with the **Deutsch–Jozsa Algorithm** —  
each with *Idea → Implementation → Code Flow → Video Link*.

Should I start with **Part 1 – Classical XOR (classical_xor.py)** next?
