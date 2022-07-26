using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse.MachineLearning;


public interface IClassifier
{
    
}

public interface IClassifier<TClassifierConfig>
{

    Tensor<Variable> Classify(Tensor<double> input);

    void TrainClassifier(Tensor<double> input, Tensor<int> labels, TClassifierConfig config);

}