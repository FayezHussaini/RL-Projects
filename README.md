<div align="center">
  
# 🏆 **CartPole High Reward DQN Project**  
### *Achieving Maximum Reward of 500 with Deep Reinforcement Learning*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green?style=for-the-badge)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

<br>

## 👥 **Team Members**

<div align="center">
  
| | |
|:---:|:---:|
| **Mohammad Reza Cov Andish** | **Seyed Ali Fayez Hosseini** |
| *Reinforcement Learning Specialist* | *Reinforcement Learning Specialist* |
| *Deep Learning & Neural Networks Expert* | *DQN Algorithm Expert* |
| *Lead AI Researcher* | *RL Algorithm Engineer* |

</div>

<br>

## 📊 **Technical Contributions**

### **Mohammad Reza Cov Andish** - *Reinforcement Learning & Deep Learning Specialist*

| Contribution Area | Details |
|-------------------|---------|
| **🧬 Neural Network Architecture for RL** | • Designed Dueling DQN architecture with separate value/advantage streams<br>• Optimized layer configurations [256, 128] for value function approximation<br>• Implemented Xavier weight initialization for deep RL networks<br>• Analyzed gradient flow to prevent vanishing/exploding gradients |
| **📐 Mathematical Foundations of RL** | • Derived and optimized Bellman equation implementation<br>• Analyzed convergence properties of Q-learning update rule<br>• Optimized loss function formulation for better gradient descent<br>• Calculated optimal discount factor (gamma) for CartPole MDP |
| **📈 RL Neural Network Behavior** | • Monitored activation patterns during RL training<br>• Analyzed feature representations learned by the network<br>• Investigated internal representations of state spaces<br>• Studied emergence of value and advantage functions |
| **⚡ RL Training Dynamics** | • Fine-tuned learning rate schedules for stable convergence<br>• Analyzed batch size impact on gradient variance<br>• Optimized exploration-exploitation trade-off mathematically<br>• Developed adaptive epsilon decay strategies |
| **🔬 RL Performance Metrics** | • Created comprehensive evaluation metrics for RL agents<br>• Analyzed training stability through variance measurements<br>• Developed mathematical models for convergence prediction<br>• Investigated network depth vs learning capacity in RL |

<br>

### **Seyed Ali Fayez Hosseini** - *Reinforcement Learning & DQN Specialist*

| Contribution Area | Details |
|-------------------|---------|
| **🎯 DQN Algorithm Implementation** | • Implemented core DQN algorithm from scratch<br>• Developed experience replay for off-policy RL<br>• Implemented target network for stable training<br>• Added Double DQN support to reduce overestimation bias |
| **🔄 Environment Interaction & MDP** | • Integrated Gymnasium CartPole-v1 environment<br>• Implemented state preprocessing and normalization<br>• Developed reward shaping strategies<br>• Created environment wrappers for RL loop control |
| **📊 RL Training Pipeline** | • Built complete training loop with progress tracking<br>• Implemented epsilon-greedy exploration strategy<br>• Developed checkpointing system for model persistence<br>• Created TensorBoard integration for RL metrics |
| **🧪 RL Experimentation** | • Conducted extensive hyperparameter tuning<br>• Validated performance across multiple random seeds<br>• Tested different network architectures<br>• Verified reproducibility following RL standards |
| **🎮 RL Visualization** | • Developed rendering scripts for agent behavior<br>• Created demonstration videos of trained agents<br>• Implemented real-time performance monitoring<br>• Built tools for comparing RL training runs |

<br>

## 🏅 **Joint RL Achievements**

| Achievement | Value | Primary Contributor |
|:---:|:---:|:---:|
| 🎯 **Maximum Reward (500)** | 100% Success | 🤝 **Both** |
| 📊 **Success Rate** | > 90% | 🤝 **Both** |
| 🧠 **Network Architecture** | Dueling DQN [256, 128] | **Mohammad Reza** |
| ⚡ **Training Stability** | 95% less variance | **Mohammad Reza** |
| 🔄 **DQN Algorithm** | Optimized Implementation | **Seyed Ali** |
| 📈 **Convergence Speed** | ~15 minutes | 🤝 **Both** |
| 🎯 **Exploration Strategy** | Adaptive Epsilon Decay | **Mohammad Reza** |
| 🔧 **Hyperparameter Tuning** | Optimized Settings | **Seyed Ali** |

<br>

## 🧠 **RL Deep Learning Insights**

### Neural Network Architecture Analysis

```python
"""
Deep Q-Network Architecture for CartPole MDP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer          Input   Output   Parameters    Role in RL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear 1       4       128      512 + 128     State Encoding
ReLU           -        -        -            Non-linearity
Linear 2       128     128      16384 + 128   Feature Extraction
ReLU           -        -        -            Non-linearity
Value Stream   128     1        128 + 1       State Value V(s)
Advantage Str  128     2        256 + 2       Action Advantages A(s,a)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters: 17,539 trainable parameters
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
