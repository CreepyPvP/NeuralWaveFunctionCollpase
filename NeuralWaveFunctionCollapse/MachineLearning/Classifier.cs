using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public interface IClassifier
{

    int Classify(Tensor input);

    void Train(Tensor input, DataContainer<int> labels);

}