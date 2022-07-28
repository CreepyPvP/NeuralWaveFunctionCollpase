using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Benchmark;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.Math.Optimisation;





public interface IOptimiser
{

    void Minimize(Variable fun, Variable[] adjustable, ITrainingBenchmark benchmark);

    void Minimize(Variable fun, ITrainingBenchmark benchmark);

}