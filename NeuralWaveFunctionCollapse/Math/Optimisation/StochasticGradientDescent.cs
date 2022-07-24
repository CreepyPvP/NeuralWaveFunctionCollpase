using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;



public struct SgdConfig
{
    public double LearnRate { get; set; }
    
    public int Iterations { get; set; }
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
    
}