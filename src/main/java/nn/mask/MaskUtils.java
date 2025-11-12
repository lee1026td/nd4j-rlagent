package nn.mask;

import tensor.Tensor;

public class MaskUtils {
    public static Tensor dropoutMaskLike(Tensor X, double dropProb) {
        if(dropProb <= 0.0) return Tensor.ones(X.shape());
        if(dropProb >= 1.0) return Tensor.zeros(X.shape());

        double keep = 1.0 - dropProb;

        return Tensor.randomBernoulli(keep, X.shape()).div(keep);
    }
}
