namespace NeuralWaveFunctionCollapse.Benchmark;

public class BenchmarkIteration
{

    private readonly long _duration;
    private readonly bool _success;

    public BenchmarkIteration(long duration, bool success)
    {
        _duration = duration;
        _success = success;
    }

    public long GetDuration()
    {
        return _duration;
    }

}