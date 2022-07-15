using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.MachineLearning.RandomForest;

public class RandomForestClassifier
{

    // INDEX
    private DataContainer<TreeClassifier> _treeStorage;
    
    // INDEX x DATA_INDEX
    private DataContainer<int>[] _indexStorage;

    /*
     *  data: INDEX x DATAPOINT
     *  labels: INDEX
     */
    public void Train(Tensor data, DataContainer<int> labels, int treeCount, int seed)
    {
        if (data.GetShape().GetDimensionality() <= 1)
            throw new Exception("Cant train on 1-dimensional data");

        var random = new SeededRandom(seed);
        
        var paramCount = data.GetShape().Size(1);
        var treeParamCount = (int) System.Math.Sqrt(paramCount);
        
        var dataPoints = data.ToArray();

        _indexStorage = 
            GenerateParamCombinations(
                    treeParamCount, 
                    Shape.Sub(data.GetShape(), 1),
                    treeCount, 
                    random)
            .ToArray();
        _treeStorage = new DataContainer<TreeClassifier>(Shape.Of(treeCount));
        
        for (var i = 0; i < treeCount; i++)
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

    public int Predict(Tensor input)
    {
        // TODO check input
        
        if (_treeStorage == null)
            throw new Exception("Random forest is still untrained");

        for (var i = 0; i < _treeStorage.GetShape().GetSizeAt(0); i++)
        {
            var tree = _treeStorage.GetValue(i);

            var trimmedData = input.ByIndexContainer(_indexStorage[i]);

            var prediction = tree.Predict(trimmedData);
        }
    }
    
}