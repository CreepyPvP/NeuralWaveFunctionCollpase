using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public interface IClassifier
{

    Tensor Classify(Tensor input);

    void Train(Tensor input, DataContainer<int> labels, int classCount);

}