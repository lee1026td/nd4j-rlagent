import gymnasium as gym
from py4j.java_gateway import JavaGateway

gateway = JavaGateway()

env = gym.make('CartPole-v1')

#create an instance of the DQN agent
agent = gateway.entry_point.getAgent()
