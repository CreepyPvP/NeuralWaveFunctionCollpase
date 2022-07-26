using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public interface IWaveFunctionModel
{

    bool Impacts(int collapseX, int collapseY, int posX, int posY);

    Tensor<double> CalculateDistribution(int x, int y, Tensor<int> collapsed, Tensor<double> additionalData);

}