# bell-state-rl
# Learning to Entangle: Bell State via Reinforcement Learning

This project explores how simple Q-learning can be used to discover a quantum circuit that creates the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.

## Structure

- `random_search/`: Randomly generates quantum circuits and evaluates fidelity.
- `q_learning/`: Reinforcement learning (Q-learning) agent that learns to entangle.
- `utils/`: Shared fidelity calculation utility.

## Dependencies

```bash
pip install -r requirements.txt
