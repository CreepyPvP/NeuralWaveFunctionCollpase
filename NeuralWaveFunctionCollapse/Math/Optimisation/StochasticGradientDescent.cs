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
    
    
    public void Minimize(Variable fun, Variable[] adjustable)
    {
        for (var i = 0; i < _config.Iterations; i++)
        {
            
            var derivatives = fun.Derive();
            foreach (var variable in adjustable)
            {
                var derivative = derivatives[variable];
                
                variable.Set(variable.Value() - _config.LearnRate * derivative);
            }
            
        }
    }

    public void Minimize(Variable fun)
    {
        for (var i = 0; i < _config.Iterations; i++)
        {
            var derivatives = fun.Derive();
            foreach (var param in derivatives.Keys)
            {
                if (!param.IsIdentity()) continue;
                
                param.Set(param.Value() - _config.LearnRate * derivatives[param]);
            }
        }
    }
    
}