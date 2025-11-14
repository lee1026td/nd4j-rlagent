import gymnasium as gym
from py4j.java_gateway import JavaGateway
from py4j.java_collections import ListConverter
import numpy as np
import time

# Gym environment
env = gym.make("CartPole-v1", render_mode="human")

gateway = JavaGateway()
converter = ListConverter()
client = gateway._gateway_client
entry = gateway.entry_point

# Get an instance of the agent
agent = entry.getAgent()

num_episodes = 100

def to_jlist(target) :
    arr = np.asarray(target, dtype=np.float64).ravel()
    py_list = [float(x) for x in arr]
    java_list = converter.convert(py_list, client)

    return java_list

def to_2djlist(target) :
    arr = np.asarray(target, dtype=np.float64)
    rows = []

    for row in arr :
        py_row = [float(x) for x in row]
        java_row = converter.convert(py_row, client)
        rows.append[java_row]

    java_2d = converter.convert(rows, client)

    return java_2d

# Train loops
for ep in range(num_episodes) :
    obs, _ = env.reset()
    done, trunc = False, False

    total_reward = 0.0

    while not (done or trunc) :
        java_state = entry.tensorFromList(to_jlist(obs))

        action = agent.act(java_state)

        next_obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        java_nstate = entry.tensorFromList(to_jlist(next_obs))

        agent.store(java_state, int(action), float(reward), java_nstate, bool(done or trunc))

        agent.learn()

        obs = next_obs

    print(f"[Episode {ep:03d}] Return = {total_reward:.1f}")

# Test the agent's performance
obs, _ = env.reset()
done, trunc = False, False

while not (done or trunc) :
        env.render()
        action = agent.act(entry.tensorFromList(to_jlist(obs)))
        next_obs, reward, done, _, _ = env.step(action)
        obs = next_obs
        time.sleep(0.1)

env.close()   

    



