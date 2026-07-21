import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import gym
except ImportError:
    print("Gym is not installed. Please install it using 'pip install gym[box2d]'")
    exit()

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'")
    exit()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # Use target network to evaluate future rewards (Double DQN style approximation)
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0]))
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            states[i] = state[0]
            targets[i] = target_f[0]
            
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    print("--------------------------------------------------")
    print("Lunar Lander RL Agent Training (DQN)")
    print("--------------------------------------------------")
    
    # Initialize the LunarLander environment
    # Note: Requires box2d to be installed (pip install gym[box2d] or box2d-py)
    try:
        env = gym.make('LunarLander-v2')
    except Exception as e:
        print("Failed to load LunarLander-v2. Please ensure Box2D is installed: pip install gym[box2d]")
        return
        
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 300
    batch_size = 64
    scores = []
    
    print("\nStarting Training...")
    print(f"Total Episodes: {episodes}")
    
    for e in range(episodes):
        # Reset the environment at the beginning of each episode
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, state_size])
        
        score = 0
        for time in range(1000): # max steps per episode
            # Agent decides action
            action = agent.act(state)
            
            # Environment steps
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
                done = done or truncated
            else:
                next_state, reward, done, _ = step_result
                
            next_state = np.reshape(next_state, [1, state_size])
            score += reward
            
            # Agent remembers the experience
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                # Update target model occasionally
                agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {score:.2f}, Epsilon: {agent.epsilon:.2f}")
                scores.append(score)
                break
                
            # Train the agent continuously during the episode
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
    print("\nTraining Complete.")
    
    # Plotting the scores
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Lunar Lander DQN Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score (Total Reward)')
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold (200)')
    plt.legend()
    plt.grid(True)
    plt.savefig('lunar_lander_training_history.png', bbox_inches='tight')
    print("Training history plot saved as 'lunar_lander_training_history.png'.")

if __name__ == "__main__":
    main()
