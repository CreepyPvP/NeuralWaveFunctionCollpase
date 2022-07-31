using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public interface ICollapseHandler
{

    int Collapse(Tensor<double> probabilities, Tensor<int> collapsed, Tensor<double> input, int x, int y);

}