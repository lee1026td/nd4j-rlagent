package algos;

import algos.dqn.DQNAgent;
import tensor.Tensor;

import java.util.List;

public class PyEnvEntryPoint {
    private Agent agent;

    public PyEnvEntryPoint(Agent agent) {
        this.agent = agent;
    }

    public Agent getAgent() {
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
