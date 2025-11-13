package algos.dqn;

import nn.activation.Activation;
import nn.core.Module;
import nn.core.Parameter;
import nn.initializer.HeNormal;
import nn.initializer.XavierNormal;
import nn.layers.Linear;
import nn.optimizer.Optimizer;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class DQN implements Module {

    private final List<Linear> layers;
    private final Activation hiddenActivation;

    public DQN(int stateSize, int actionSize, Activation hiddenActivation, int... hiddenSize) {
        assert(hiddenSize.length > 0);

        layers = new ArrayList<>();
        this.hiddenActivation = hiddenActivation;
        int layerNum = hiddenSize.length - 1;

        layers.add(new Linear(stateSize, hiddenSize[0], new XavierNormal(), new HeNormal(), true));
        for(int i=0; i<layerNum; i++) {
            layers.add(new Linear(hiddenSize[i], hiddenSize[i+1], new XavierNormal(), new HeNormal(), true));
        }
        layers.add(new Linear(hiddenSize[layerNum], actionSize, new XavierNormal(), new HeNormal(), true));
    }


    @Override
    public Tensor forward(Tensor X, boolean training) {
        Tensor Y = X;
        for(Linear layer : layers) {
            Y = layer.forward(Y, training);
            Y = hiddenActivation.forward(Y);
        }

        return Y;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        Tensor dX = (scale == 1.0) ? dY : dY.mul(scale);

        for(Linear layer : layers) {
            dX = layer.calcGradients(dX, accumulate, scale);
            dX = hiddenActivation.backward(dX);
        }

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        for(Linear layer : layers) {
            layer.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> parameters = new ArrayList<>();
        for(Linear layer : layers) {
            parameters.addAll(layer.parameters());
        }

        return parameters;
    }

    @Override
    public void zeroGrad() {
        for(Linear layer : layers) {
            layer.zeroGrad();
        }
    }

    public void copyFrom(DQN source) {
        List<Parameter> sourceParams = source.parameters();

        assert(sourceParams.size() == parameters().size());

        for(int i=0,j=0;i<layers.size();i++,j+=2) {
            layers.get(i).setParameters(List.of(sourceParams.get(j), sourceParams.get(j+1)));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("DQN {\n");
        for(Linear layer : layers) {
            sb.append("\t" + layer.toString() + "\n");
        }
        sb.append("}\n");

        return sb.toString();
    }
}
