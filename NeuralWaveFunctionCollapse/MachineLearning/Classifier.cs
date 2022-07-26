using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;


public interface IClassifier
{
    
}

public interface IClassifier<TConfiguration>
{

    Tensor<double> Classify(Tensor<double> input);

    void Train(Tensor<double> input, Tensor<int> labels, TConfiguration configuration);

}