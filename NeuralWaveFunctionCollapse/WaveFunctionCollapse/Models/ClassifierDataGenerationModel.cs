using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

public class ClassifierDataGenerationModel: IWaveFunctionModel
{


    private readonly int _radius;
    

    public ClassifierDataGenerationModel(int radius)
    {
        _radius = radius;
    }
    
    public bool Impacts(int collapseX, int collapseY, int posX, int posY)
    {
        return System.Math.Abs(collapseX - posX) <= _radius && System.Math.Abs(collapseY - posY) <= _radius;
    }

    public Tensor<double> CalculateDistribution(int x, int y, Tensor<int> collapsed, Tensor<double> additionalData)
    {
        throw new NotImplementedException();
    }
    
}