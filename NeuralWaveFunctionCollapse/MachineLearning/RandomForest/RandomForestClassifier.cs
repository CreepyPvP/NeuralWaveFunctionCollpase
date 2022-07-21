using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.MachineLearning.RandomForest;

public class RandomForestClassifier: IClassifier
{

    // INDEX
    private DataContainer<TreeClassifier>? _treeStorage;
    
    // INDEX x DATA_INDEX
    private DataContainer<int>[]? _indexStorage;

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
    public void Train(Tensor data, DataContainer<int> labels, int classCount)
    {
        if (data.GetShape().GetDimensionality() <= 1)
            throw new Exception("Cant train on 1-dimensional data");

        _outputClasses = classCount;
        
        var paramCount = data.GetShape().Size(1);
        var treeParamCount = (int) System.Math.Sqrt(paramCount);
        
        var dataPoints = data.ToArray();

        _indexStorage = 
            GenerateParamCombinations(
                    treeParamCount, 
                    Shape.Sub(data.GetShape(), 1),
                    _treeCount, 
                    _random)
            .ToArray();
        _treeStorage = new DataContainer<TreeClassifier>(Shape.Of(_treeCount));
        
        for (var i = 0; i < _treeCount; i++)
        {
            var tree = new TreeClassifier();
            _treeStorage.SetValue(tree, i);

            var trainData =
                dataPoints.Select(dataPoint => dataPoint.ByIndexContainer(_indexStorage[i])).ToArray();
            
            tree.Train(trainData, labels);
        }
    }

    private DataContainer<int> GenerateParamCombinations(int length, Shape shape, int count, SeededRandom random)
    {
        var result = new DataContainer<int>(Shape.Of(count, length));

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

    public Tensor Classify(Tensor input)
    {
        // TODO check input
        
        if (_treeStorage == null || _indexStorage == null)
            throw new Exception("Random forest is still untrained");
        
        var output = new Tensor(Shape.Of(_outputClasses));

        var treeCount = _treeStorage.GetShape().GetSizeAt(0);
        
        for (var i = 0; i < treeCount; i++)
        {
            var tree = _treeStorage.GetValue(i);
            var trimmedData = input.ByIndexContainer(_indexStorage[i]);

            var prediction = tree.Predict(new Tensor(trimmedData));
            
            output.SetValue(output.GetValue(prediction) + 1.0 / treeCount, prediction);
        }

        return output;
    }
    
}