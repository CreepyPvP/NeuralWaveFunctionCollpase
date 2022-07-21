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

    // width x height x inputLayerCount
    public void Build(Tensor input, int classCount)
    {
        if (input.GetShape().GetDimensionality() != 3 || input.GetShape().GetSizeAt(2) <= 0)
            throw new Exception("Invalid input data");

        var width = input.GetShape().GetSizeAt(0);
        var height = input.GetShape().GetSizeAt(1);

        // training size x neuron count x inputs
        var trainingData = new Tensor(Shape.Of(width * height, GetKernelSize(_radius), input.GetShape().GetSizeAt(2)));
        var labels = new DataContainer<int>(Shape.Of(width * height));

        var i = 0;
        
        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                labels.SetValue((int) input.GetValue(x, y, 0), i);

                var index = 0;
            
                for (var dX = 0; dX < _radius; dX++)
                {
                    for (var dY = 0; dY < _radius; dY++)
                    {
                        // TODO: add samples with uncollapsed indices ( -1 )
                        trainingData.Copy(GetDataAt(x, y, input), trainingData.GetShape().GetIndex(i, index, 0));
                        index++;
                    }
                }

                i++;
            }
        }
        
        _classifier.Train(trainingData, labels, classCount);
    }
    
    public bool Impacts(int collapseX, int collapseY, int posX, int posY)
    {
        return System.Math.Abs(collapseX - posX) <= _radius && System.Math.Abs(collapseY - posY) <= _radius;
    }

    public Tensor CalculateDistribution(int collapsedX, int collapsedY, DataContainer<int> collapsed, Tensor additionalData)
    {
        // Pos x property
        var request = new Tensor(Shape.Of(GetKernelSize(_radius), additionalData.GetShape().GetSizeAt(2) + 1));

        var index = 0;
        
        for (var dX = 0; dX < _radius; dX++)
        {
            for (var dY = 0; dY < _radius; dY++)
            {
                var x = collapsedX + dX;
                var y = collapsedY + dY;
                
                request.Copy(GetDataAt(x, y, collapsed, additionalData), request.GetShape().GetIndex(index, 0));
                index++;
            }
        }

        return _classifier.Classify(request);
    }


    private Tensor GetDataAt(int x, int y, DataContainer<int> collapsed, Tensor input)
    {
        var outputSize = input.GetShape().GetSizeAt(2);
        var result = new Tensor(Shape.Of(outputSize + 1));
        
        if (x < 0 || y < 0 ||
            x >= collapsed.GetShape().GetSizeAt(0) ||
            y >= collapsed.GetShape().GetSizeAt(1))
        {
            // Default out of boundaries
            result.SetValue(1, 0);
            
            for (var i = 0; i < outputSize; i++)
            {
                result.SetValue(0, i + 1);
            }
            
            return result;
        }
        
        result.SetValue(collapsed.GetValue(x, y), 0);

        for (var i = 0; i < outputSize; i++)
        {
            result.SetValue(input.GetValue(x, y, i), i + 1);
        }
        
        return result;
    }


    private Tensor GetDataAt(int x, int y, Tensor input)
    {
        var outputSize = input.GetShape().GetSizeAt(2);
        var result = new Tensor(Shape.Of(outputSize));
        
        if (x < 0 || y < 0 ||
            x >= input.GetShape().GetSizeAt(0) ||
            y >= input.GetShape().GetSizeAt(1))
        {
            for (var i = 0; i < outputSize; i++)
            {
                result.SetValue(0, i);
            }
            
            return result;
        }
        
        for (var i = 0; i < outputSize; i++)
        {
            result.SetValue(input.GetValue(x, y, i), i);
        }
        
        return result;
    }

    private int GetKernelSize(int radius)
    {
        return (2 * radius + 1) * (2 * radius + 1) - 1;
    }
    
}