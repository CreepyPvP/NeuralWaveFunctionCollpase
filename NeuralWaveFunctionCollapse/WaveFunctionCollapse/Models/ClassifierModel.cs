using NeuralWaveFunctionCollapse.MachineLearning;
using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

public class ClassifierModel: IWaveFunctionModel
{


    private readonly int _radius;
    
    private IClassifier _classifier;

    
    public ClassifierModel(IClassifier classifier, int radius)
    {
        _classifier = classifier;
        _radius = radius;
    }

    public void Build()
    {
        var trainingSize = 0;
        var trainingData = new Tensor(Shape.Of(trainingSize, GetKernelSize(_radius),));
    }
    
    public bool Impacts(int collapseX, int collapseY, int posX, int posY)
    {
        return System.Math.Abs(collapseX - posX) <= 1 && System.Math.Abs(collapseY - posY) <= 1;
    }

    public Tensor CalculateDistribution(int x, int y, DataContainer<int> collapsed, Tensor additionalData)
    {
        throw new NotImplementedException();
    }


    private int GetKernelSize(int radius)
    {
        return (2 * radius + 1) * (2 * radius + 1) - 1;
    }
    
}