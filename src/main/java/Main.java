import algos.Agent;
import algos.PyEnvEntryPoint;
import algos.dqn.DQN;
import algos.dqn.DQNAgent;
import algos.dqn.Experience;
import nn.activation.GELU;
import nn.activation.ReLU;
import nn.core.Parameter;
import nn.initializer.*;
import nn.layers.*;
import nn.loss.*;
import nn.normalizer.*;
import nn.optimizer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;
import py4j.ClientServer;
import tensor.Nd4jInit;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import py4j.GatewayServer;

public class Main {
    public static void main(String[] args) {
        Nd4jInit.configure();

        /* Hyper parameters */
        int stateSize = 4;
        int actionSize = 2;
        double lr = 1e-3;
        double gamma = 0.99;
        int bufferCapacity = 50000;
        int batchSize = 64;
        double epsStart = 1.0;
        double epsEnd = 0.05;
        int epsDecaySteps = 3000;

        /* Agent */
        Agent dqnAgent = new DQNAgent(stateSize, actionSize, lr, gamma, bufferCapacity, batchSize, epsStart, epsEnd, epsDecaySteps);

        /* Entrypoint of the java process gateway */
        PyEnvEntryPoint pyEnv = new PyEnvEntryPoint(dqnAgent);

        /* Gateway instance */
        GatewayServer gatewayServer = new GatewayServer(pyEnv);
        /* Start the server */
        gatewayServer.start();
        System.out.println("Gateway Server Started");

    }
}
