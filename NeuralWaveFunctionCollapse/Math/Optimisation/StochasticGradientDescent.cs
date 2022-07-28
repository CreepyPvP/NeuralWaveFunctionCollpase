using System.Runtime.InteropServices.ComTypes;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Benchmark;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;



public struct SgdConfig
{
    public double LearnRate;

    public int Iterations;
}


public class StochasticGradientDescentOptimiser: IOptimiser
{


    private readonly SgdConfig _config;


    public StochasticGradientDescentOptimiser(SgdConfig config)
    {
        _config = config;
    }


    public void Minimize(Variable fun, Variable[] adjustable, ITrainingBenchmark? benchmark)
    {
        for (var i = 0; i < _config.Iterations; i++)
        {

            var values = new Dictionary<Variable, double>();
            var derivatives = fun.Derive(values);
            foreach (var variable in adjustable)
            {
                var derivative = derivatives[variable];
                
                variable.Set(variable.Value() - _config.LearnRate * derivative);
                
                if(benchmark != null)
                    benchmark.PushResult(values[fun]);
            }
            
        }
    }

    public void Minimize(Variable fun, ITrainingBenchmark? benchmark)
    {
        for (var i = 0; i < _config.Iterations; i++)
        {
            var values = new Dictionary<Variable, double>();
            var derivatives = fun.Derive(values);
            
            foreach (var param in derivatives.Keys)
            {
                if (!param.IsIdentity()) continue;
                
                param.Set(param.Value() - _config.LearnRate * derivatives[param]);
                
                if(benchmark != null)
                    benchmark.PushResult(values[fun]);
            }
        }
    }
    
}