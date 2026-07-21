# Cart-Pole RL Agent Training (DQN)

## Objective
To train an autonomous Reinforcement Learning (RL) agent to balance a pole on a moving cart using the classic `CartPole-v1` environment from OpenAI Gym. The agent is trained using a Deep Q-Network (DQN).

## Environment Details
- **CartPole-v1**: A pendulum is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart (pushing it left or right).
- **State Space**: An array of 4 values representing:
  1. Cart Position
  2. Cart Velocity
  3. Pole Angle
  4. Pole Angular Velocity
- **Action Space**: 2 discrete actions (0: push left, 1: push right).
- **Goal**: Keep the pole balanced upright for as long as possible. The episode ends if the pole falls over (angle > 12 degrees) or the cart moves off-screen.

## Methodology
1. **Deep Q-Network (DQN)**: 
   - A Neural Network (acting as the Q-function approximator) is built using Keras, featuring two hidden `Dense` layers of 24 neurons each, using ReLU activation.
   - The network takes the 4-dimensional state as input and outputs the Q-values for the 2 possible actions.
2. **Experience Replay**: 
   - Instead of learning from sequential steps immediately (which are highly correlated), the agent stores its experiences `(state, action, reward, next_state, done)` in a memory buffer.
   - During training, it samples a random minibatch of 32 experiences to train the neural network, stabilizing the learning process.
3. **Epsilon-Greedy Strategy**: 
   - The agent starts by taking completely random actions (`epsilon = 1.0`) to explore the environment.
   - Over time, the exploration rate gradually decays (`epsilon_decay = 0.995`), shifting the agent towards exploiting its learned Q-values to maximize the score.

## Expected Results
As the agent progresses through the 200 episodes, you will observe the score (time steps survived) initially hovering around 10-20. As the epsilon decays and the neural network learns, the score will drastically improve, eventually reaching the maximum 500 steps. 

## How to Run
Ensure you have Gym, TensorFlow, and Matplotlib installed:
```bash
pip install gym tensorflow matplotlib
```
Execute the training script:
```bash
python dqn_cartpole.py
```
A plot named `cartpole_training_history.png` will be generated, visualizing the agent's learning curve.
