using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class RandomCollapseHandler: ICollapseHandler
{

    private readonly SeededRandom _random;

    
    public RandomCollapseHandler()
    {
        _random = new SeededRandom(0);
    }
    
    public RandomCollapseHandler(int seed)
    {
        _random = new SeededRandom(seed);
    }
    
    public int Collapse(Tensor<double> probabilities, Tensor<int> collapsed, Tensor<double> input, int x, int y)
    {
        return _random.NextIndex(probabilities, true, true);
    }
    
}