﻿using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning.RandomForest;

public class TreeClassifier
{


    private ITreeElement _root;
    

    public int Predict(Tensor input)
    {
        if (_root == null)
            throw new Exception("Tree is not built");

        return 0;
    }

    /*
     *  data: INDEX x DATAPOINT
     *  labels: INDEX
     */
    public void Train(Tensor data, DataContainer<int> labels)
    {
        if (data.GetShape().GetSizeAt(0) != labels.GetShape().GetSizeAt(0) ||
            labels.GetShape().GetDimensionality() != 1)
            throw new Exception("Invalid data passed");
        
        _root = new TreeDecisionElement();
        
        _root.Build(data.ToArray(), labels.ToArray(), 1);
    }

    public void Save()
    {
        throw new Exception("Not implemented");
    }

    public void Load()
    {
        throw new Exception("Not implemented");
    }
    
    
}


interface ITreeElement
{
    
    /*
     *  data: INDEX x DATAPOINT
     *  labels: INDEX
     */
    void Build(DataContainer<double>[] data, DataContainer<int>[] labels, double parentEntropy);

    int Predict(Tensor input);

}


class TreeDecisionElement: ITreeElement
{

    private double _splitPoint;
    
    // the position of the information that is used to split
    private int[]? _dataPosition = null;

    private ITreeElement _trueBranch;

    private ITreeElement _falseBranch;
    

    public void Build(DataContainer<double>[] data, DataContainer<int>[] labels, double parentEntropy)
    {
        if (data.Length <= 0) 
            throw new Exception("Trying to train a binary filter on 0 data points");
        var shape = data[0].GetShape();

        double minTrueEntropy = 0;
        double minFalseEntropy = 0;

        shape.ForEach(position =>
        {
            foreach (var datapoint in data)
            {
                var splitPoint = datapoint.GetValue(position);

                var trueLabels = new List<int>();
                var falseLabels = new List<int>();

                for (var i = 0; i < data.Length; i++)
                {
                    if (Apply(data[i], position, splitPoint))
                    {
                        trueLabels.Add(labels[i].GetValue(0));
                    }
                    else
                    {
                        falseLabels.Add(labels[i].GetValue(0));
                    }
                }

                var trueEntropy = CalculateEntropy(trueLabels);
                var falseEntropy = CalculateEntropy(falseLabels);

                if ((trueEntropy + falseEntropy < minTrueEntropy + minFalseEntropy) || _dataPosition == null)
                {
                    _dataPosition = position;
                    _splitPoint = splitPoint;
                    minTrueEntropy = trueEntropy;
                    minFalseEntropy = falseEntropy;
                }
            }
        });

        _trueBranch = parentEntropy - minTrueEntropy > 0.0 ? new TreeDecisionElement() : new TreeLeafElement();
        _falseBranch = parentEntropy - minFalseEntropy > 0.0 ? new TreeDecisionElement() : new TreeLeafElement();

        List<DataContainer<double>> trueData = new();
        List< DataContainer<int>> trueLabels = new();
        List<DataContainer<double>> falseData = new();
        List< DataContainer<int>> falseLabels = new();
        for (var i = 0; i < data.Length; i++)
        {
            if (Apply(data[i], _dataPosition!, _splitPoint))
            {
                trueData.Add(data[i]);
                trueLabels.Add(labels[i]);
            }
            else
            {
                falseData.Add(data[i]);
                falseLabels.Add(labels[i]);
            }
        }
        _trueBranch.Build(trueData.ToArray(), trueLabels.ToArray(), minTrueEntropy);
        _falseBranch.Build(falseData.ToArray(), falseLabels.ToArray(), minFalseEntropy);
    }

    public int Predict(Tensor input)
    {
        if (_dataPosition == null)
            throw new Exception("Trying to predict using untrained tree");


        return Apply(input, _dataPosition, _splitPoint) ? _trueBranch.Predict(input) : _falseBranch.Predict(input);
    }


    private bool Apply(DataContainer<double> tensor, int[] splitPos, double splitValue)
    {
        return tensor.GetValue(splitPos) <= splitValue;
    }

    private static double CalculateEntropy(IReadOnlyList<int> labels)
    {
        Dictionary<int, int> classToAmount = new();
        for (var i = 0; i < labels.Count; i++)
        {
            var current = classToAmount.ContainsKey(labels[i]) ? classToAmount[labels[i]] : 0;
            classToAmount[labels[i]] = current + 1;
        }

        double entropy = 0;
        foreach (var key in classToAmount.Keys)
        {
            var probability = (double)key / labels.Count;

            entropy -= probability * System.Math.Log(probability);
        }

        return entropy;
    }
    
}


class TreeLeafElement: ITreeElement
{

    private int _label;
    public void Build(DataContainer<double>[] data, DataContainer<int>[] labels, double parentEntropy)
    {
        Dictionary<int, int> classToAmount = new();
        for (var i = 0; i < labels.Length; i++)
        {
            var current = classToAmount.ContainsKey(labels[i].GetValue(0)) ? classToAmount[labels[i].GetValue(0)] : 0;
            classToAmount[labels[i].GetValue(0)] = current + 1;
        }

        var maxCount = 0;
        foreach (var label in classToAmount.Keys)
        {
            if (classToAmount[label] <= maxCount) continue;
            _label = label;
            maxCount = classToAmount[label];
        }
    }

    public int Predict(Tensor input)
    {
        return _label;
    }
}