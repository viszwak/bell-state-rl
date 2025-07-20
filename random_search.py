import cirq
import numpy as np
import matplotlib.pyplot as plt

# Define the target Bell state: (|00⟩ + |11⟩) / sqrt(2)
target_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex64)

# Fidelity function (ensure both inputs are flattened and length 4)
def fidelity(state1, state2):
    state1 = np.asarray(state1).flatten()
    state2 = np.asarray(state2).flatten()
    if state1.shape != (4,) or state2.shape != (4,):
        raise ValueError(f"State shape mismatch: {state1.shape} vs {state2.shape}")
    return np.abs(np.vdot(state1, state2)) ** 2

# Define qubits
q0, q1 = cirq.LineQubit.range(2)

# Gate actions
ACTIONS = [
    lambda c: c.append(cirq.H(q0)),
    lambda c: c.append(cirq.H(q1)),
    lambda c: c.append(cirq.X(q0)),
    lambda c: c.append(cirq.X(q1)),
    lambda c: c.append(cirq.CNOT(q0, q1)),
    lambda c: c.append(cirq.I(q0)),
    lambda c: c.append(cirq.I(q1))
]
action_names = ["H(q0)", "H(q1)", "X(q0)", "X(q1)", "CNOT", "I(q0)", "I(q1)"]

# Setup
simulator = cirq.Simulator()
num_trials = 500
num_steps = 3
fidelities = []
best_fidelity = 0
best_circuit = None
best_actions = []

# Random search loop
for _ in range(num_trials):
    circuit = cirq.Circuit()
    actions_taken = []
    for _ in range(num_steps):
        idx = np.random.choice(len(ACTIONS))
        actions_taken.append(idx)
        ACTIONS[idx](circuit)

    result = simulator.simulate(circuit)
    output_state = result.final_state_vector

    try:
        f = fidelity(output_state, target_state)
    except ValueError as e:
        print("Skipping invalid state:", e)
        continue

    fidelities.append(f)

    if f > best_fidelity:
        best_fidelity = f
        best_circuit = circuit
        best_actions = [action_names[i] for i in actions_taken]

# Plot fidelity over trials
plt.figure(figsize=(10, 4))
plt.plot(fidelities, label="Fidelity")
plt.axhline(1.0, linestyle='--', color='green', label="Target Fidelity = 1.0")
plt.xlabel("Trial")
plt.ylabel("Fidelity")
plt.title("Random Search: Fidelity to Bell State Over Trials")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print best result
print("Best Circuit:\n", best_circuit)
print("Gate sequence:", best_actions)
print(f"Best Fidelity: {best_fidelity:.4f}")
