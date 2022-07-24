using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;

public interface IOptimiser<in TConfiguration>
{

    void Minimize(Variable fun, Variable[] adjustable, TConfiguration configuration);

}