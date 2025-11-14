package algos;

import algos.dqn.DQNAgent;
import tensor.Tensor;

import java.util.List;

public class PyEnvEntryPoint {
    private DQNAgent agent;

    public PyEnvEntryPoint() {
        agent = new DQNAgent(
            4, 2, 1e-3, 0.99, 50000, 64, 1.0, 0.05, 3000
        );
    }

    public DQNAgent getAgent() {
        return agent;
    }

    public Tensor tensorFromList(List<Double> list) {
        double[] arr = list.stream().mapToDouble(Double::doubleValue).toArray();

        return Tensor.from(arr, arr.length);
    }

    public Tensor tensorFromList2D(List<List<Double>> list) {
        double[][] arr = list.stream().map(rowList -> rowList.stream().mapToDouble(Double::doubleValue).toArray()).toArray(double[][]::new);
        return Tensor.from(arr);
    }

}
