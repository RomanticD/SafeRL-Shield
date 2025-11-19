# SafeRL-Shield

A neural network-based safety shield framework for safe reinforcement learning in hazardous environments.

## Overview

SafeRL-Shield implements a multi-class risk classification system that acts as a protective shield for Q-learning agents navigating environments with safety hazards. The project demonstrates how deep learning can proactively prevent reinforcement learning agents from taking dangerous actions while maintaining task performance.

## Key Features

**Multi-Class Risk Assessment**
- 4-level danger classification (immediate hazard, 1-step away, 2-steps away, safe)
- Trained neural network predicts risk for each possible action
- Real-time action filtering based on configurable risk thresholds

**Adaptive Safety Shield**
- Intervenes only when necessary to block dangerous actions
- Selects safest alternative while maximizing expected reward
- Tunable safety-performance trade-off via threshold parameter θ

**Comprehensive Evaluation**
- Compares baseline unsafe Q-learning with shielded variants
- Tracks safety violations, interventions, and cumulative rewards
- Visualizes danger maps and confusion matrices for interpretability

## Highlights

- **Zero-violation Learning**: Shield with θ=0 achieves perfect safety (0 hazard collisions) while maintaining reward performance
- **Minimal Intervention**: Safety is guaranteed with minimal agent autonomy loss - only critical actions are overridden
- **Stratified Training**: 70/15/15 train-val-test split with class balancing ensures robust risk predictions
- **End-to-End Pipeline**: From environment setup to danger mapping, dataset generation, shield training, and safe RL integration

## Technical Stack

- **Environment**: Custom SafeGridWorld with 15 hazard cells in 10×10 grid
- **RL Algorithm**: Tabular Q-learning with ε-greedy exploration
- **Safety Model**: 2-layer fully-connected neural network (64 hidden units)
- **Framework**: TensorFlow/Keras for deep learning, NumPy for RL implementation

## Results

In 1000-episode training runs with random starts:
- **Baseline Q-Learning**: ~127 safety violations
- **Shield θ=0**: 0 violations, comparable reward, <50 interventions
- **Shield θ=2**: 0 violations, degraded reward, excessive interventions (>500)

The optimal configuration (θ=0) demonstrates that safety and performance are not mutually exclusive when intelligent shielding is applied.

## Project Structure

```
├── sample.ipynb          # Main implementation notebook
├── data/
│   ├── best_qtable.pkl          # Best Q-learning policy
│   ├── completedataset.pkl      # Training/validation/test data
│   ├── shield_model.h5          # Trained safety shield
│   ├── shield_metrics.pkl       # Model evaluation metrics
│   └── task5_results.pkl        # Safe RL experiment results
└── README.md
```

## Quick Start

Run the cells in `sample.ipynb` sequentially:
1. **Task 1**: Initialize SafeGridWorld environment
2. **Task 2**: Train baseline Q-learning agents
3. **Task 3**: Generate danger maps and safety dataset
4. **Task 4**: Train multi-class risk classifier (shield)
5. **Task 5**: Integrate shield with Q-learning and evaluate

## Applications

This framework can be extended to:
- Robotics navigation in dynamic environments
- Autonomous vehicle control with safety constraints
- Medical treatment optimization with risk bounds
- Industrial process control with failure prevention
