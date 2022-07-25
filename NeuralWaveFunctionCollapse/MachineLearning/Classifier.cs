using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;


public interface IClassifier
{
    
}

public interface IClassifier<TConfiguration>
{

    Tensor Classify(Tensor input);

    void Train(Tensor input, DataContainer<int> labels, TConfiguration configuration);

}