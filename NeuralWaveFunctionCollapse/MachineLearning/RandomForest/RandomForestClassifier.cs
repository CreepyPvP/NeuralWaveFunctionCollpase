using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;
using NeuralWaveFunctionCollapse.Util;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

namespace NeuralWaveFunctionCollapse.MachineLearning.RandomForest;



public readonly struct RandomForestTrainingConfiguration
{

    public readonly int ClassCount { get; }

}


public class RandomForestClassifier: IWaveFunctionClassifier<RandomForestTrainingConfiguration>
{

    // INDEX
    private Tensor<TreeClassifier>? _treeStorage;
    
    // INDEX x DATA_INDEX
    private Tensor<int>[]? _indexStorage;

    private readonly SeededRandom _random;

    private readonly int _treeCount;

    private int _outputClasses;
    
    
    public RandomForestClassifier(int treeCount, int seed)
    {
        _treeCount = treeCount;
        _random = new SeededRandom(seed);
    }
    
    
    /*
     *  data: INDEX x DATAPOINT
     *  labels: INDEX
     */
    public void TrainClassifier(Tensor<double> input, Tensor<int> labels, RandomForestTrainingConfiguration configuration)
    {
        if (input.GetShape().GetDimensionality() <= 1)
            throw new Exception("Cant train on 1-dimensional data");

        _outputClasses = configuration.ClassCount;
        
        var paramCount = input.GetShape().Size(1);
        var treeParamCount = (int) System.Math.Sqrt(paramCount);
        
        var dataPoints = input.ToArray();

        _indexStorage = 
            GenerateParamCombinations(
                    treeParamCount, 
                    Shape.Sub(input.GetShape(), 1),
                    _treeCount, 
                    _random)
            .ToArray();
        _treeStorage = new Tensor<TreeClassifier>(Shape.Of(_treeCount));
        
        for (var i = 0; i < _treeCount; i++)
        {
            var tree = new TreeClassifier();
            _treeStorage.SetValue(tree, i);

            var trainData =
                dataPoints.Select(dataPoint => dataPoint.ByIndexContainer(_indexStorage[i])).ToArray();
            
            tree.Train(trainData, labels);
        }
    }

    public void Build(int kernelSize, int outputClasses, int inputDimensions)
    {
        
    }

    public void Build(int kernelSize, int outputDimensions, int inputDimensions, string file, IoManager ioManager)
    {
        throw new NotImplementedException();
    }

    public void Save(string file, IoManager ioManager)
    {
        throw new NotImplementedException();
    }

    private Tensor<int> GenerateParamCombinations(int length, Shape shape, int count, SeededRandom random)
    {
        var result = new Tensor<int>(Shape.Of(count, length));

        var indexStorage = new int[shape.Size()];
        for (var i = 0; i < indexStorage.Length; i++)
        {
            indexStorage[i] = i;
        }

        for (var i = 0; i < count; i++)
        {
            random.Shuffle(indexStorage);
            for (var j = 0; j < length; j++)
            {
                result.SetValue(indexStorage[j], i, j);
            }
        }
        
        return result;
    }

    public Tensor<Variable> Classify(Tensor<double> input)
    {
        // TODO check input
        
        if (_treeStorage == null || _indexStorage == null)
            throw new Exception("Random forest is still untrained");
        
        var output = new Tensor<Variable>(Shape.Of(_outputClasses));

        var treeCount = _treeStorage.GetShape().GetSizeAt(0);
        
        for (var i = 0; i < treeCount; i++)
        {
            var tree = _treeStorage.GetValue(i);
            var trimmedData = input.ByIndexContainer(_indexStorage[i]);

            var prediction = tree.Predict(new Tensor<double>(trimmedData));
            
            // TODO: set output.GetValue default!
            
            output.SetValue(output.GetValue(prediction) + (1.0 / treeCount), prediction);
        }

        return output;
    }

}