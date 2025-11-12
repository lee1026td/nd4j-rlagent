package algos;

import algos.dqn.DQNAgent;

public class PyEnvEntryPoint {
    private DQNAgent agent;

    public PyEnvEntryPoint() {
        agent = new DQNAgent(
            4, 2, 1e-3, 0.99, 1000, 64, 1.0, 0.05, 10000
        );
    }

    public DQNAgent getAgent() {
        return agent;
    }

}
