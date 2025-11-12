package algos.dqn;

import algos.Agent;
import nn.activation.ReLU;
import nn.loss.Loss;
import nn.loss.MSELoss;
import nn.optimizer.Adam;
import nn.optimizer.Optimizer;
import tensor.Tensor;

import java.util.Random;

public class DQNAgent implements Agent {

    private final int stateSize, actionSize;
    private final int batchSize;

    private DQN evalNet, targetNet;
    private final ReplayBuffer buffer;
    private final Optimizer optimizer;
    private final Loss mseLoss;

    private final double gamma;
    private double epsStart, epsEnd, eps;
    private int epsDecaySteps;
    private int epsStep;

    private final Random rand = new Random();

    public DQNAgent(int stateSize,
                    int actionSize,
                    double lr,
                    double gamma,
                    int bufferCapacity,
                    int batchSize,
                    double epsStart,
                    double epsEnd,
                    int epsDecaySteps) {

        this.stateSize = stateSize;
        this.actionSize = actionSize;

        this.batchSize = batchSize;

        evalNet = new DQN(stateSize, actionSize, new ReLU(), 128, 128, 128);
        targetNet = new DQN(stateSize, actionSize, new ReLU(), 128, 128, 128);

        targetNet.copyFrom(evalNet);

        buffer = new ReplayBuffer(bufferCapacity);

        optimizer = new Adam(lr, 0.9, 0.99, 1e-8);
        mseLoss = new MSELoss();

        this.gamma = gamma;
        this.eps = this.epsStart = epsStart;
        this.epsEnd = epsEnd;
        this.epsDecaySteps = epsDecaySteps;
        this.epsStep = 0;
    }

    // Epsilon-Greedy policy
    public int act(Tensor state) {
        double prob = rand.nextDouble();

        if(prob <= eps) {                                           // Epsilon-Greedy
            return rand.nextInt(actionSize);                        // Exploration
        } else {
            Tensor pred = evalNet.forward(state, false);
            return pred.argmax(0).getInt(0);            // Exploitation
        }
    }

    // Store the Experience instance into ReplayBuffer : (s, a, r, s', done)
    public void store(Tensor state, int action, double reward, Tensor nextState, boolean done) {
        buffer.store(new Experience(state, action, reward, nextState, done));
    }

    // Epsilon-Greedy scheduling with decaying epsilon
    private void stepEpsilon() {
        if(epsStep < epsDecaySteps) {
            double t = (++epsStep) / (double) epsDecaySteps;
            eps = epsStart + t * (epsEnd - epsStart);
        } else {
            eps = epsEnd;
        }
    }

    public boolean learn() {
        stepEpsilon();

        Experience[] batch = buffer.sample(batchSize);

        evalNet.zeroGrad();

        Tensor states, nextStates;
        int[] actions = new int[batchSize];

        for(int i=0;i<batchSize;i++) {
            Experience e = batch[i];
        }

        return true;
    }
}
