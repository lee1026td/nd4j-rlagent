package algos.dqn;

import algos.Agent;
import nn.activation.ReLU;
import nn.loss.Loss;
import nn.loss.MSELoss;
import nn.optimizer.Adam;
import nn.optimizer.Optimizer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
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

    private final int warmupSteps;        // Buffer warmups
    private final int targetSyncEvery;    // Target network sync interval
    private int trainSteps;

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

        this.warmupSteps = Math.max(1000, batchSize * 5);
        this.targetSyncEvery = 1000;
        this.trainSteps = 0;
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
        public void store(Tensor state, Integer action, Double reward, Tensor nextState, Boolean done) {
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
        if(buffer.size() < warmupSteps) {
            return false;
        }

        trainSteps++;

        // Sample transition (experience) batches of size batchSize from ReplayBuffer
        Experience[] batch = buffer.sample(batchSize);

        Tensor[] states = new Tensor[batchSize];
        Tensor[] nextStates = new Tensor[batchSize];
        int[] actions = new int[batchSize];         // [batchSize]
        double[] rewards = new double[batchSize];   // [batchSize]
        boolean[] dones = new boolean[batchSize];   // [batchSize]

        for(int i=0;i<batchSize;i++) {
            Experience e = batch[i];
            states[i] = e.state;
            nextStates[i] = e.nextState;
            actions[i] = e.action;
            rewards[i] = e.reward;
            dones[i] = e.done;
        }

        Tensor s = Tensor.vstack(states);            // [batchSize, stateSize]
        Tensor sNext = Tensor.vstack(nextStates);    // [batchSize, stateSize]

        // Evaluation from samples : Q(s,.)
        Tensor qEval = evalNet.forward(s, true);    // [batchSize, actionSize]

        // Get Q(s,a)
        Tensor[] tmp = new Tensor[batchSize];
        for(int i=0;i<batchSize;i++) {
            tmp[i] = qEval.get(i, actions[i]);
        }
        Tensor qSA = Tensor.vstack(tmp);        // [batchSize]

        // Next state evaluation from Target Network : Q'(s',.)
        Tensor qNext = targetNet.forward(sNext, false); // [batchSize, actionSize]
        // max_a' Q'(s',a')
        Tensor maxQNext = qNext.max(1, false);    // [batchSize]

        // Compute TD target : y = r + gamma * (1 - done) * max_a' Q'(s',a')
        double[] targetArr = new double[batchSize];
        for(int i=0;i<batchSize;i++) {
            targetArr[i] = rewards[i] + gamma * (1.0 - (dones[i] ? 1.0 : 0.0)) * maxQNext.getDouble(i);
        }

        Tensor target = Tensor.from(targetArr, batchSize);

        // Compute loss
        double loss = mseLoss.forward(qSA, target);

        Tensor dLoss = mseLoss.backward();

        Tensor dY = Tensor.zeros(batchSize, actionSize);
        for(int i=0;i<batchSize;i++) {
            int a = actions[i];
            double g = dLoss.getDouble(a);
            dY.set(g, i, a);
        }

        System.out.println(dY);

        // Backpropagate dY to evalNet
        evalNet.backward(dY);
        evalNet.update(optimizer);
        evalNet.zeroGrad();

        syncTargetNetwork();



        return true;
    }

    private void syncTargetNetwork() {
        if(trainSteps % targetSyncEvery == 0) {
            targetNet.copyFrom(evalNet);
        }
    }


}
