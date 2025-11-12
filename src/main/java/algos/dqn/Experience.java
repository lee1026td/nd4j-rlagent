package algos.dqn;

import tensor.Tensor;

public class Experience {

    public final Tensor state;
    public final int action;
    public final double reward;
    public final Tensor nextState;
    public final boolean done;

    public Experience(Tensor state, int action, double reward, Tensor nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}
