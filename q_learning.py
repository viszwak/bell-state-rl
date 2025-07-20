import cirq
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Target Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
target_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex64)

# Fidelity function
def fidelity(state1, state2):
    return np.abs(np.vdot(state1, state2)) ** 2

# Define qubits
q0, q1 = cirq.LineQubit.range(2)

# Available gate actions
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

# Q-learning parameters
num_actions = len(ACTIONS)
num_steps = 3
q_table = defaultdict(lambda: np.zeros(num_actions))
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Training setup
episodes = 1000
fidelities = []
best_fidelity = 0
best_actions = []

simulator = cirq.Simulator()

for episode in range(episodes):
    state = tuple()
    circuit = cirq.Circuit()
    circuit.append([cirq.I(q0), cirq.I(q1)])  # ⬅️ Force 2-qubit system always
    actions_taken = []

    for step in range(num_steps):
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(num_actions)
        else:
            action_idx = np.argmax(q_table[state])
        actions_taken.append(action_idx)
        ACTIONS[action_idx](circuit)
        state += (action_idx,)

    result = simulator.simulate(circuit)
    output_state = result.final_state_vector
    reward = fidelity(output_state, target_state)
    fidelities.append(reward)

    if reward > best_fidelity:
        best_fidelity = reward
        best_actions = actions_taken[:]

    # Q-table update
    for i in range(num_steps):
        s = tuple(actions_taken[:i])
        a = actions_taken[i]
        next_s = tuple(actions_taken[:i+1])
        max_q = np.max(q_table[next_s])
        q_table[s][a] += alpha * (reward + gamma * max_q - q_table[s][a])

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Smoothing
def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode="valid")

# Plot Fidelity over Episodes
plt.figure(figsize=(10, 5))
plt.plot(fidelities, color="gray", alpha=0.3, label="Raw Fidelity")
plt.plot(moving_average(fidelities), color="blue", label="Smoothed Fidelity (20-ep MA)")
plt.axhline(1.0, color="green", linestyle="--", label="Target Fidelity = 1.0")
plt.xlabel("Episode")
plt.ylabel("Fidelity with Bell State")
plt.title("Q-Learning: Fidelity Over Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show best circuit discovered
print("\nBest learned circuit:")
circuit = cirq.Circuit()
circuit.append([cirq.I(q0), cirq.I(q1)])
for idx in best_actions:
    ACTIONS[idx](circuit)
print(circuit)
print(f"Fidelity: {best_fidelity:.4f}")
