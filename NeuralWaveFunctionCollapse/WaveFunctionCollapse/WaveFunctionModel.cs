using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public interface IWaveFunctionModel
{

    bool Impacts(int collapseX, int collapseY, int posX, int posY);

    Tensor CalculateDistribution(int x, int y, DataContainer<int> collapsed, Tensor additionalData);

}