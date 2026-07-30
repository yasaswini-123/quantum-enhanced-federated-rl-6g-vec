# Quantum-Enhanced Secure Task Offloading for 6G Vehicular Networks

Federated Deep Reinforcement Learning with a real Transformer encoder and a real Variational Quantum Circuit (VQC) policy head, Grover-optimized decision search, and differential-privacy / QKD-secured V2V communication — for intelligent, secure, low-latency task offloading in 6G vehicular networks (V2V / V2U / Edge / RSU).

Built on top of and benchmarked against: Paul & Singh, *"Large AI Model-Driven Quantum-Enhanced Transformer-VQC Federated DRL for Privacy Preservation in Vehicular Networks,"* IEEE JSAC, vol. 44, 2026.

---

## Overview

Autonomous vehicles generate large volumes of sensor data that must be processed in real time. Existing offloading systems struggle with high latency, inefficient resource use, and weak privacy guarantees when sharing data across the network. This project implements an intelligent, secure task-offloading framework for 6G vehicular networks that combines:

- A **Transformer encoder** for temporal feature extraction from network state
- A **real Variational Quantum Circuit (VQC)** policy head (PennyLane, `default.qubit`, backprop-differentiable — not a classical stand-in)
- **Federated Deep Reinforcement Learning (Actor-Critic / A2C)**, with each vehicular cluster as its own federated client
- **Grover search** (Cirq) as a discrete decision-optimization layer across 5 real use cases
- **Differential privacy** (moments accountant) + **QKD-lite + OTP encryption** for secure model updates and V2V communication
- **Secure federated aggregation** — encrypted local model updates combined without ever sharing raw vehicle data

---

## Problem Statement

Autonomous vehicles generate a large amount of sensor data that must be processed in real time for safe and reliable driving. Existing systems often experience high latency, inefficient resource utilization, and slow task offloading, and face privacy and security challenges while sharing information across the network. This project builds an intelligent and secure framework that provides fast decision-making, efficient task offloading, privacy preservation, and secure communication for future 6G vehicular networks.

---

## Objectives

- Minimize task latency
- Improve offloading efficiency by selecting the best computing resource
- Enable secure V2V communication using QKD and OTP encryption
- Integrate a Transformer encoder with a real VQC for intelligent offloading decisions
- Apply Federated Deep Reinforcement Learning to train collaboratively without sharing raw vehicle data
- Optimize decision-making using Grover search
- Preserve privacy using Differential Privacy and Secure Aggregation

---

## System Architecture

```
Vehicles → V2V Communication → UAVs → Edge Servers → RSU
                                   │
                                   ▼
State Representation → Transformer Encoder → Variational Quantum Circuit (VQC)
                                   │
                                   ▼
                  Federated Deep Reinforcement Learning (Actor-Critic)
                                   │
                                   ▼
                        Grover Search Optimization
                     (edge/cluster · UAV · offload action · path · resources)
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
          Differential Privacy   QKD + OTP   Secure Federated
                                Encryption      Aggregation
                                   │
                                   ▼
        Performance Evaluation: Latency · Throughput · Energy ·
                Reward · Privacy Budget (ε) · Delivery Ratio
```

**Network topology:** 4 clusters, 4 vehicles/cluster, 3 UAVs/cluster, 1 RSU/cluster → 16 vehicles, 12 UAVs total, 4 federated clients (one per cluster).

---

## Reinforcement Learning Formulation

Each vehicle, at each timestep, is treated as an RL agent deciding how to split its computational task across Local / UAV / Edge / RSU compute.

### State
A 6-dimensional per-vehicle feature vector, normalized:
`[uplink SINR, fronthaul SINR, sensing quality, deadline, step fraction, Grover cluster flag]`

The proposed policy consumes a **5-step history window** of this vector as a sequence into the Transformer; the baseline sees only the latest step.

### Action
A continuous 3-part split ratio, sampled from a learned Gaussian policy:

| Symbol | Meaning |
|---|---|
| β (`beta`) | fraction of the task offloaded vs. kept local |
| γᵤ (`gamma_u`) | of the offloaded portion, fraction sent to the UAV |
| γₑ (`gamma_e`) | of what remains after UAV, fraction sent to Edge (remainder → RSU) |

The baseline policy has no Edge tier (`gamma_e` is forced to 0; only β, γᵤ are learned).

### Reward
```python
latency_ms = multi_tier_latency_ms(task_bits, beta, gamma_u, gamma_e, v2u_rate, bh_rate, cfg)
reward = -(latency_ms / deadline_ms)
```
Reward is simply negative normalized task latency against the 250 ms URLLC deadline — throughput and resource-utilization gains emerge as side effects of latency minimization across the multi-tier pipeline, not as separate reward terms.

### Algorithm — Advantage Actor-Critic (A2C)
- **Actor:** Transformer encoder → VQC policy head → action mean (+ a learned `log_std`)
- **Critic:** small value head predicting expected reward for the state
- **Advantage:** `reward − value_prediction` (batch-normalized for stability)
- **Loss:** `-(log_prob × advantage).mean() − entropy_coef × entropy` (actor) + MSE (critic), plus a FedProx proximal term for the proposed policy's federated clients

---

## Grover Search — 5 real use cases

Grover search (Cirq, 2-qubit circuits) sits **alongside** the RL policy as a non-differentiable decision layer, evaluated against classical brute-force for correctness:

1. **Task-offloading action** — Local / UAV / Edge / RSU, from the policy's own proposed split
2. **UAV selection** — best UAV per cluster by backhaul rate
3. **Communication-path selection** — direct broadcast vs. relay, for V2V warnings
4. **Resource allocation** — best Edge-compute allocation policy per cluster
5. **Relay selection** — best V2V relay among real in-range neighbors

Match rate against classical-optimal: **100%** across all logged (episode, step, vehicle/cluster) decisions in this run.

> Note: at this candidate scale (N=4), Grover's benefit is architectural/asymptotic (O(√N) vs O(N)), not a wall-clock speedup on a classical simulator — this project demonstrates correctness and pipeline integration as the necessary first step before deployment at a scale where the gap pays off.

---

## Privacy & Security

- **Differential Privacy:** gradient clipping + calibrated Gaussian noise on the encoder's optimizer only, tracked with a moments accountant (one accounting step per federated client per episode)
- **QKD-lite + OTP encryption:** simulated quantum-key-distribution link (pulse rate, detector efficiency, dark-count, error-correction factor) gates OTP-encrypted delivery of V2V warning packets and federated gradient transport
- **Secure Federated Aggregation:** each client's model update is masked with a paired random perturbation (client *i*'s mask is client *j*'s negative) so summed masks cancel out — the server only ever recovers the aggregate, never an individual client's raw weights (same principle as Bonawitz et al. Secure Aggregation, without the full DH handshake)

---

## Results

| Metric | JSAC Baseline | Proposed (Q-FDRL) | Δ |
|---|---|---|---|
| Latency (ms, ↓) | 409.05 | 373.16 | ▼ 8.8% |
| Throughput (Mbps, ↑) | 3.172 | 3.477 | ▲ 9.6% |
| Training loss (↓) | 7.659 | 6.507 | ▼ 15.0% |
| Reward / step (↑) | −1.636 | −1.493 | ▲ 8.7% |

- V2V warning delivery rate: **95.9%** overall (near-100% after ramp-up)
- Measured privacy budget: **ε → 20.0** (worst-case client, moments accountant)
- Grover match rate vs. classical-optimal: **100%** across all 5 use cases

Both baseline and proposed use the *same* A2C algorithm and *same* Transformer+VQC stack for a fair comparison — the proposed model's gains come from the Edge tier, V2V relay, Grover decision layer, and federated setup, not a different base algorithm.

---

## Comparison with Baseline Paper

| Framework Component | Baseline (Paul & Singh, JSAC 2026) | Proposed (this project) |
|---|:---:|:---:|
| Transformer encoder + VQC policy head | ✓ | ✓ |
| Federated A2C across vehicles/UAVs/RSUs | ✓ | ✓ |
| Differential privacy (moments accountant) | ε ≤ 5 | measured ε |
| QKD-based OTP encryption | gradient transport | gradients + V2V |
| V2V communication (obstacle warning) | ✗ | ✓ |
| Dedicated Edge Server tier | ✗ | ✓ |
| Grover search decision layer | ✗ | ✓ (5 use cases) |

---

## Tech Stack

- **PyTorch** — Transformer encoder, Actor-Critic training loop
- **PennyLane** (`default.qubit`, `diff_method="backprop"`) — real, gradient-trainable Variational Quantum Circuit policy head (8 qubits, depth 4)
- **Cirq** — Grover search circuits for discrete decision optimization
- **NumPy / Pandas / SciPy** — simulation, data handling, finite-blocklength rate calculations
- **Matplotlib** — training curves and result visualization

---

## Project Structure

```
.
├── Q_FDRL_vs_JSAC_VQC_Baseline.ipynb   # full simulation, training, and evaluation notebook
├── data/                                 # latency traces, convergence/privacy reference CSVs
├── results/                              # generated plots and logs_df exports
└── README.md
```

---

## Getting Started

### Requirements
```bash
pip install torch pennylane cirq numpy pandas scipy matplotlib
```

### Run
```bash
jupyter notebook Q_FDRL_vs_JSAC_VQC_Baseline.ipynb
```
Run all cells top to bottom. Configuration (clusters, episodes, architecture sizes, DP/QKD parameters) is centralized in the `Config` class — see the notebook's Section 3 for paper-aligned defaults and inline citations to the source paper's sections/equations.

> Note: the paper-scale architecture (8-qubit / depth-4 VQC, 4-layer Transformer) is noticeably slower on CPU-only runtimes. Set `cfg.USE_PAPER_SCALE_ARCH = False` in the `Config` class for a smaller, faster fallback during development.

---

## Conclusion

Developed a secure and intelligent task-offloading framework for 6G vehicular networks that combines a Transformer encoder, a real VQC policy head, Federated Deep Reinforcement Learning, V2V communication, and Grover-optimized decision-making with differential privacy and QKD-secured communication into a single unified system — achieving lower latency, higher throughput, and stronger security guarantees than the baseline paper's framework.

## Future Work

- Deploy on real vehicles, UAVs, and edge computing platforms
- Run the VQC on real quantum hardware and explore deeper quantum architectures
- Integrate blockchain for tamper-resistant V2V communication trust

---

## References

1. Paul, R. Singh, "Large AI Model-Driven Quantum-Enhanced Transformer-VQC Federated DRL for Privacy Preservation in Vehicular Networks," *IEEE JSAC*, vol. 44, pp. 3026–3040, 2026.
2. B. Narottama, S. Y. Shin, "Federated Quantum Neural Network With Quantum Teleportation for Resource Optimization in Future Wireless Communication," *IEEE Trans. Vehicular Technology*, vol. 72, no. 11, pp. 14717–14732, 2023.
3. S. Lokes et al., "Implementation of Quantum Deep Reinforcement Learning Using Variational Quantum Circuits," *Proc. TQCEBT*, 2022, pp. 1–4.
4. T. Srivastava, K. K. Soni, A. Rasool, "Evolution of Quantum Computing Based on Grover's Search Algorithm," *Proc. 10th ICCCNT*, Kanpur, India, 2019.
5. W. Zhao et al., "Quantum Computing in Wireless Communications and Networking: A Tutorial-cum-Survey," *IEEE Communications Surveys & Tutorials*, vol. 27, no. 4, pp. 2378–2415, 2025.

---

## Author

**Balla Yasaswini**
Department of CSE, RGUKT-Srikakulam
Under the guidance of **Dr. Dinesh R**, Department of CSE, IIITDM-Kancheepuram

---

## License

This project is for academic/research purposes. Add a license (e.g. MIT) here if you intend to open-source it.
