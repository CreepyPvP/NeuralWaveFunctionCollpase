using System.Runtime.Versioning;
using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.MachineLearning;
using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

public class ClassifierModel<TTrainConfiguration>: IWaveFunctionModel
{


    private readonly int _radius;
    private readonly int _outputClasses;
    
    private IWaveFunctionClassifier<TTrainConfiguration> _classifier;

    
    public ClassifierModel(IWaveFunctionClassifier<TTrainConfiguration> classifier, int radius, int outputClasses)
    {
        _classifier = classifier;
        _radius = radius;
        _outputClasses = outputClasses;
    }


    public void Build(Tensor<double>[] inputs, int inputDimensionality, TTrainConfiguration configuration, int periods, String path = null, IoManager ioManager = null)
    {
        foreach (var input in inputs)
        {
            if (input.GetShape().GetDimensionality() != 3 || input.GetShape().GetSizeAt(2) != inputDimensionality)
                throw new Exception("Invalid input data");
        }


        var collapseHandlers = new LoggerCollapseHandler[inputs.Length];
        for (var i = 0; i < inputs.Length; i++)
        {
            collapseHandlers[i] = new LoggerCollapseHandler(inputs[i], GetNeighborhoodAt, GetKernelSize(_radius));
        }

        _classifier.Build(GetKernelSize(_radius), _outputClasses, inputDimensionality);
        
        for (var period = 0; period < periods; period++)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
                var collapseHandler = collapseHandlers[i];
         
                var width = inputs[i].GetShape().GetSizeAt(0);
                var height = inputs[i].GetShape().GetSizeAt(1);
                
                var grid = new Grid(width, height, _outputClasses, this, collapseHandler);
                grid.Collapse();
            
                Console.WriteLine("Training Period: " + period + ", Map: " + i + " ---------------------------------");
                _classifier.TrainClassifier(collapseHandler.DecisionData, collapseHandler.DecisionLabels, configuration);

                collapseHandler.ResetHead();   
                
                if(path != null && ioManager != null)
                    Save(path + "-" + period + ".json", ioManager);
            }
        }
    }

    public bool Impacts(int collapseX, int collapseY, int posX, int posY)
    {
        return System.Math.Abs(collapseX - posX) <= _radius && System.Math.Abs(collapseY - posY) <= _radius;
    }

    public Tensor<double> CalculateDistribution(int collapsedX, int collapsedY, Tensor<int> collapsed, Tensor<double> additionalData)
    {
        var request = GetNeighborhoodAt(collapsedX, collapsedY, collapsed, additionalData);

        return _classifier.Classify(request).Evaluate();
    }


    private Tensor<double> GetDataAt(int x, int y, Tensor<int> collapsed, Tensor<double> input)
    {
        var outputSize = input.GetShape().GetSizeAt(2);
        var result = new Tensor<double>(Shape.Of(outputSize + 1));
        
        if (x < 0 || y < 0 ||
            x >= collapsed.GetShape().GetSizeAt(0) ||
            y >= collapsed.GetShape().GetSizeAt(1))
        {
            // Default out of boundaries
            result.SetValue(-2, 0);
            
            for (var i = 0; i < outputSize; i++)
            {
                result.SetValue(-2, i + 1);
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


    public Tensor<double> GetNeighborhoodAt(int collapsedX, int collapsedY, Tensor<int> collapsed, Tensor<double> input)
    {
        // Pos x property
        var request = new Tensor<double>(Shape.Of(GetKernelSize(_radius), input.GetShape().GetSizeAt(2) + 1));

        var index = 0;
        
        for (var dX = -_radius; dX <= _radius; dX++)
        {
            for (var dY = -_radius; dY <= _radius; dY++)
            {
                if(dX == 0 && dY == 0) continue;

                var x = collapsedX + dX;
                var y = collapsedY + dY;
                
                request.Copy(GetDataAt(x, y, collapsed, input), request.GetShape().GetIndex(index, 0));
                index++;
            }
        }

        return request;
    }

    public void Save(string file, IoManager ioManager)
    {
        _classifier.Save(file, ioManager);
    }
    
    
    private int GetKernelSize(int radius)
    {
        return (2 * radius + 1) * (2 * radius + 1) - 1;
    }
    
    
}