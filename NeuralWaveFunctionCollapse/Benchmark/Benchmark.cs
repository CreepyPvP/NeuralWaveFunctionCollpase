namespace NeuralWaveFunctionCollapse.Benchmark;

public class Benchmark
{

    private readonly Action _action;

    private readonly int _iterations;

    private readonly List<BenchmarkIteration> _results = new();


    public Benchmark(Action action, int iterations)
    {
        _action = action;
        _iterations = iterations;
    }

    public void Run()
    {
        for (var i = 0; i < _iterations; i++)
        {
            var start = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            bool success = true;
            
            try
            {
                _action.Invoke();
            }
            catch (Exception e)
            {
                success = false;
            }
            var end = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            var duration = end - start;
            
            _results.Add(new BenchmarkIteration(duration, success));
        }
    }


    public double AvgTime => TotalTime / (double) _iterations;

    public double TotalTime => _results.Sum(result => result.GetDuration()) / (double) 1000;
    
    
}