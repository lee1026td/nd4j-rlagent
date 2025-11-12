package nn.loss;

import tensor.Tensor;

public class MSELoss implements Loss {

    private Tensor diff;
    private int N;

    @Override
    public double forward(Tensor preds, Tensor tgts) {
        return forward(preds, tgts, null);
    }

    @Override
    public double forward(Tensor preds, Tensor tgts, Tensor mask) {
        this.diff = preds.sub(tgts);

        this.N = preds.size();
        Tensor squared = diff.mul(diff);

        int ndim = squared.ndim();

        Tensor loss = squared;
        for(int i = 0; i < ndim - 1; i++) {
            loss = loss.sum(i, false);
        }

        return loss.getInt(0) * (1.0 / N);
    }

    @Override
    public Tensor backward() {
        if(this.diff != null && this.N != 0) {
            double scale = 2.0f / this.N;
            return this.diff.mul(scale);
        }
        return null;
    }
}
