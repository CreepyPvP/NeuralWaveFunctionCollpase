using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;





public interface IOptimiser
{

    void Minimize(Variable fun, Variable[] adjustable);

    void Minimize(Variable fun);

}