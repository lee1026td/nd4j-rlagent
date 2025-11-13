import algos.PyEnvEntryPoint;
import algos.dqn.DQN;
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

        GatewayServer gatewayServer = new GatewayServer(new PyEnvEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");

    }
}
