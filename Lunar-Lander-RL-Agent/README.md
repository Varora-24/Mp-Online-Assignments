# Lunar Lander RL Agent Training (DQN)

## Objective
To train an autonomous Reinforcement Learning (RL) agent to safely land a spacecraft on the moon's surface using the `LunarLander-v2` environment from OpenAI Gym. The agent is trained using an advanced Deep Q-Network (DQN) with a Target Network.

## Environment Details
- **LunarLander-v2**: A classic rocket trajectory optimization problem. The landing pad is always at coordinates (0,0). The agent is rewarded for moving toward the pad and landing safely, and penalized for crashing or using fuel.
- **State Space**: An array of 8 continuous values representing:
  1. X coordinate
  2. Y coordinate
  3. X velocity
  4. Y velocity
  5. Angle
  6. Angular velocity
  7. Left leg contact (boolean)
  8. Right leg contact (boolean)
- **Action Space**: 4 discrete actions:
  0. Do nothing
  1. Fire left orientation engine
  2. Fire main engine
  3. Fire right orientation engine
- **Goal**: Achieve a score of 200 or higher.

## Methodology
1. **Deep Q-Network (DQN)**: 
   - A Neural Network acts as the Q-function approximator, built with Keras. Due to the slightly larger state and action space compared to CartPole, this network uses two hidden `Dense` layers of 64 neurons each with ReLU activation.
2. **Target Network**:
   - To stabilize training, a secondary "Target" network is maintained. It evaluates the future Q-values during the loss calculation and its weights are updated to match the main network at the end of every episode. This reduces moving target oscillations.
3. **Experience Replay**: 
   - The agent stores up to 100,000 experiences in a memory buffer and samples random minibatches (size 64) to break correlation between sequential steps.
4. **Epsilon-Greedy Strategy**: 
   - The agent balances exploration (random actions) and exploitation (using learned Q-values) with an exponentially decaying epsilon (`decay = 0.995`).

## How to Run
Ensure you have Gym (with Box2D support), TensorFlow, and Matplotlib installed:
```bash
pip install gym[box2d] tensorflow matplotlib
```
*(Note on Windows: You may need to install `swig` before installing `box2d`)*

Execute the training script:
```bash
python dqn_lunar_lander.py
```
A plot named `lunar_lander_training_history.png` will be generated, visualizing the agent's reward progression over the episodes toward the threshold of 200.
