using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;



public readonly struct SgdConfig
{
    public readonly double LearnRate { get; }
    
    public readonly int Iterations { get; }
}


public class StochasticGradientDescentOptimiser: IOptimiser<SgdConfig>
{
    
    public void Minimize(Variable fun, Variable[] adjustable, SgdConfig configuration)
    {
        for (var i = 0; i < configuration.Iterations; i++)
        {
            
            var derivatives = fun.Derive();
            foreach (var variable in adjustable)
            {
                var derivative = derivatives[variable];
                
                variable.Set(variable.Value() - configuration.LearnRate * derivative);
            }
            
        }
    }

    public void Minimize(Variable fun, SgdConfig configuration)
    {
        for (var i = 0; i < configuration.Iterations; i++)
        {
            var derivatives = fun.Derive();
            foreach (var param in derivatives.Keys)
            {
                if (!param.IsIdentity()) continue;
                
                param.Set(param.Value() - configuration.LearnRate * derivatives[param]);
            }
        }
    }
    
}