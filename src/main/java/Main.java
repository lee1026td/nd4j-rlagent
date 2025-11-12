import algos.PyEnvEntryPoint;
import algos.dqn.DQN;
import algos.dqn.Experience;
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

//        DQN n1 = new DQN(4, 2, new ReLU(), 16, 16);
//        DQN n2 = new DQN(4, 2, new ReLU(), 16, 16);
//
//        n1.copyFrom(n2);
//
//        List<Parameter> n1params = n1.parameters();
//        List<Parameter> n2params = n2.parameters();
//
//        for(int i=0;i<n1params.size();i++) {
//            System.out.println(i+"\t"+n1params.get(i).getData());
//            System.out.println(i+"\t"+n2params.get(i).getData());
//        }

//        GatewayServer gatewayServer = new GatewayServer(new PyEnvEntryPoint());
//        gatewayServer.start();
//        System.out.println("Gateway Server Started");
//

        Tensor a = Tensor.from(new double[] {1, 2, 3, 4}, 4);
        Tensor b = Tensor.from(new double[] {5, 6, 7, 8}, 4);

        Experience[] arr = new Experience[]{
                new Experience(a, 1, 1.0, a, false),
                new Experience(b, 2, 1.3, b, false)
        };

        INDArray[] ff = new INDArray[arr.length];
        for(int i=0;i<arr.length;i++){
            ff[i] = arr[i].state.getNDArray();
        }

        Tensor states = new Tensor(Nd4j.concat(0, ff));

        System.out.println(states.reshape(arr.length, 4));

    }
}
