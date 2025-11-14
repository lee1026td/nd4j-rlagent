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

/*
        int batchSize = 16;
        int hidden = 128;
        int s = 4;
        int a = 2;

        Tensor k = Tensor.rand(batchSize, s);

        Linear l1 = new Linear(s, hidden, new XavierNormal(), new HeNormal(), true);

        Tensor h1 = l1.forward(k, true);

        Linear l2 = new Linear(hidden, hidden, new XavierNormal(), new HeNormal(), true);

        Tensor h2 = l2.forward(h1, true);

        Linear l3 = new Linear(hidden, a, new XavierNormal(), new HeNormal(), true);

        Tensor out = l3.forward(h2, true);
        Tensor target = Tensor.randomBernoulli(0.2, batchSize, a);

        System.out.println(out);
        System.out.println(target);

        Loss mseLoss = new MSELoss();

        double loss = mseLoss.forward(out, target);

        System.out.println(loss);

        Tensor dLoss = mseLoss.backward();
        System.out.println("dLoss : " + dLoss);

        Tensor d3 = l3.backward(dLoss);
        System.out.println("d3 : " + d3);

        Tensor d2 = l2.backward(d3);
        System.out.println("d2 : " + d2);

        Tensor d1 = l1.backward(d2);
        System.out.println("d1 : " + d1);

        Optimizer adam = new Adam(1e-3, 0.9, 0.98, 5e-5);

        l3.update(adam);
        l2.update(adam);
        l1.update(adam);*/
    }
}
