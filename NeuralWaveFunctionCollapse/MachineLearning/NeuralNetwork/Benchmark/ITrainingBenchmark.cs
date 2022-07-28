namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Benchmark;

public interface ITrainingBenchmark
{

    public void PushResult(double loss);

    public void EndEpoch();

}