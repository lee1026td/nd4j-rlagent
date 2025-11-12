package algos.dqn;

import java.util.concurrent.ThreadLocalRandom;

public class ReplayBuffer {

    private final Experience[] buffer;
    private final int maxCapacity;
    private int size;
    private int idx;

    public ReplayBuffer(int bufferCapacity) {
        buffer = new Experience[bufferCapacity];
        this.maxCapacity = bufferCapacity;
        this.size = 0;
        this.idx = 0;
    }

    public void store(Experience e) {
        buffer[idx] = e;
        idx = (idx + 1) % maxCapacity;
        size = Math.min(size + 1, maxCapacity);
    }

    public Experience[] sample(int batchSize) {
        Experience[] samples = new Experience[batchSize];
        for(int i = 0; i < batchSize; i++) {
            int k = ThreadLocalRandom.current().nextInt(size + 1);

            samples[i] = buffer[k];
        }
        return samples;
    }

    public int size() {
        return size;
    }

}
