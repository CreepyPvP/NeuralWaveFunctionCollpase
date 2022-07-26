using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;



public interface IOptimiser
{
    
}

public interface IOptimiser<in TConfiguration>: IOptimiser
{

    void Minimize(Variable fun, Variable[] adjustable, TConfiguration configuration);

    void Minimize(Variable fun, TConfiguration configuration);

}