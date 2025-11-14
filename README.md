# 🤖 ND4J RL Agents with Deep Reinforcement Learning Algorithms
Pure Java implementations of RL agent using several Deep RL algorithms  
The algorithms I want to implement are as follows:  
- DQN (Deep Q-Network) ✅
- VPG (Vanilla Policy Gradients)
- A3C (Asynchronous Actor-Critic)
- TRPO (Trust Region Policy Optimization)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradients)
## Development environments
- Java JDK 21
- Python 3.13
## Dependencies
- Py4j : https://www.py4j.org/
  - Used to connect processes between the Python environment and the Java agent.
- ND4J : https://javadoc.io/doc/org.nd4j/nd4j-api
  - Used for a more natural representation of Tensor data structures in Java.
## Settings
- All agent algorithms were written in Java, and [OpenAI Gymnasium](https://gymnasium.farama.org/) was run in Python for validating implementations.
- By default, `env.py` is set to the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment, and the Java agent is also set to match that environment.
## How to run
1. First, Run Java process to create gateway server and agent to interact with Gym environment.
2. Second, run `env.py` while the Java process is running. This will automatically start training the Agent, and after a certain number of episodes (`100` by default), you can directly check the performance based on the training results.
## Example results (From DQN)
- Early in the beginning of learning process (iteration 0-20)
  - Poor performance due to high exploration probability
<div align="center">
  <img width="50%" height="50%" alt="Image" src="https://github.com/user-attachments/assets/ae818062-983e-419a-a281-0f320c07265f" />
</div>  

- More iterations (iteration 20-80)
  - It's slightly better.
<div align="center">
  <img width="50%" height="50%" alt="Image" src="https://github.com/user-attachments/assets/29afdaff-c118-42e1-a2e0-a853dd04c5c0" />
</div>  

- Final train result
<div align="center">
  <img width="50%" height="50%" alt="Image" src="https://github.com/user-attachments/assets/c05c9eba-cfff-42b6-b831-cc2699c68149" />
</div>  
