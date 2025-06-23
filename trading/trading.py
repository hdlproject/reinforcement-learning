from trading.env import TradingEnv
from trading.nn import DQN
import numpy as np
import torch
import gymnasium as gym

episodes = 500
# how much the exploration will be conducted
epsilon = 1
# how much the future reward is valued compared to the immediate reward
gamma = 0.99
# learning rate
alpha = 0.1
q_table = {}

# number of input neurons
input_dim = 3
# number of hidden layer neurons
hidden_dim = 16
# number of output neurons (actions)
output_dim = 3

# how often the target network is updated
target_update_freq = 10


def get_state(observation):
    return (observation['price'], observation['crypto_balance'], observation['fiat_balance'])


def run():
    # initialize the environment
    env = TradingEnv()

    for ep in range(episodes):
        # reset the environment to generate the first observation
        observation, _ = env.reset()
        state = get_state(observation)
        total_reward = 0

        while True:
            # this is where you would insert your policy
            if np.random.rand() < epsilon or state not in q_table:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(next_observation)

            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            # q-learning update
            target = reward + (gamma * np.max(q_table[next_state]))
            td_error = target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

            # if the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                print(f"Episode {ep + 1}: Total reward = {total_reward:.2f}")
                break

    env.close()


def run_nn():
    total_rewards = []

    # how much the exploration will be conducted
    epsilon = 1
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # initialize the neural network
    q_net = DQN(input_dim, hidden_dim, output_dim)
    target_net = DQN(input_dim, hidden_dim, output_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=alpha)
    loss_fn = torch.nn.MSELoss()

    # initialize the environment
    env = TradingEnv()

    for ep in range(episodes):
        np_random, _ = gym.utils.seeding.np_random()

        # reset the environment to generate the first observation
        observation, _ = env.reset()
        state = get_state(observation)
        total_reward = 0

        terminated = False
        while not terminated:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # this is where you would insert your policy
            if np_random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = torch.argmax(q_values).item()

            # step (transition) through the environment with the action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(next_observation)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # q-learning update using neural network
            target = reward + (gamma * torch.max(target_net(next_state_tensor)).item() * (1 - terminated))
            target_tensor = torch.tensor(target, dtype=torch.float32)

            # calculate the loss
            current_q_value = q_net(state_tensor)[action]
            loss = loss_fn(current_q_value, target_tensor.detach())

            # calculate and record the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        # decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # update target network
        if ep % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # if the episode has ended then we can reset to start a new episode
        total_rewards.append(total_reward)
        print(f"Episode {ep + 1}: Total reward = {total_reward:.2f}")
        env.render_any(total_reward)

    np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')

    env.close()
