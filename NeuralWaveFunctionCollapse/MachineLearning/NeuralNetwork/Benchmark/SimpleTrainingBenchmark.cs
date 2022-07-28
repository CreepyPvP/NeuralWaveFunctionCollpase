namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Benchmark;

public class SimpleTrainingBenchmark: ITrainingBenchmark
{


    private double _totalLoss;
    private int _totalAttempts;

    public SimpleTrainingBenchmark()
    {
        _totalAttempts = 0;
        _totalLoss = 0;
    }
    
    public void PushResult(double loss)
    {
        loss += _totalLoss;
        _totalAttempts++;
    }
    
    public void EndEpoch()
    {
        Console.WriteLine("Avg Loss: " + _totalLoss / _totalAttempts);
        _totalLoss = 0;
        _totalAttempts = 0;
    }
    
}