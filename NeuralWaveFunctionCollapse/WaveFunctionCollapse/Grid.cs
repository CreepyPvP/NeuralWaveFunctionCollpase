using System.Diagnostics;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class Grid
{


    private readonly int _outputElements;

    private readonly Tensor<int> _output;

    // width x height x outputComplexity
    private readonly Tensor<double> _probabilities;

    // width x height x inputLayerCount
    private readonly Tensor<double> _input;

    private readonly int _width;
    private readonly int _height;

    private readonly IWaveFunctionModel _model;

    private readonly ICollapseHandler _collapseHandler;


    public Grid(int width, int height, int outputElements, IWaveFunctionModel model, ICollapseHandler collapseHandler): 
        this(width, height, outputElements, new Tensor<double>(Shape.Of(width, height, 0)), model, collapseHandler) {}

    public Grid(int width, int height, int outputElements, Tensor<double> input, IWaveFunctionModel model, ICollapseHandler collapseHandler): 
        this(width, height, outputElements, new Tensor<double>(Shape.Of(width, height, outputElements)), input, model, collapseHandler) { }

    public Grid(int width, int height, int outputElements, Tensor<double> initialStates, Tensor<double> input, IWaveFunctionModel model, ICollapseHandler collapseHandler)
    {
        _outputElements = outputElements;
        _probabilities = new Tensor<double>(Shape.Of(width, height, outputElements), 1.0);
        _input = input;

        _width = width;
        _height = height;

        _output = new Tensor<int>(Shape.Of(width, height), -1);

        _model = model;
        _collapseHandler = collapseHandler;
        
        if (initialStates.GetShape().GetSizeAt(0) != width || 
            initialStates.GetShape().GetSizeAt(1) != height ||
            initialStates.GetShape().GetSizeAt(2) != outputElements ||
            initialStates.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid initial state");
        
        if (input.GetShape().GetSizeAt(0) != width || 
            input.GetShape().GetSizeAt(1) != height ||
            input.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid input format");

        CalculateInitialDistribution();
    }


    private void CalculateInitialDistribution()
    {
        for (var x = 0; x < _width; x++)
        {
            for (var y = 0; y < _height; y++)
            {
                UpdateDistribution(x, y);
            }
        }
    }
    
    public void Collapse()
    {
        while (GetLowestEntropy(out var x, out var y)) 
        {
            var probabilities = _probabilities.Slice(2, x, y, 0);
            var collapsedElement = _collapseHandler.Collapse(probabilities, _output, _input, x, y);

            _output.SetValue(collapsedElement, x, y);
            
            PropagateCollapse(x, y);
        }
    }


    private void PropagateCollapse(int x, int y)
    {
        for (var xI = 0; xI < _width; xI++)
        {
            for (var yI = 0; yI < _height; yI++)
            {
                if(xI == x && yI == y) continue;

                if (_model.Impacts(x, y, xI, yI)) UpdateDistribution(xI, yI);
            }
        }
    }

    private bool GetLowestEntropy(out int resultX, out int resultY)
    {
        resultX = 0;
        resultY = 0;
        
        bool success = false;
        double entropy = 0;
        
        for (var x = 0; x < _width; x++)
        {
            for (var y = 0; y < _height; y++)
            {
                var localEntropy = GetEntropy(x, y);
                if ((!success || localEntropy < entropy) && _output.GetValue(x, y) == -1)
                {
                    resultX = x;
                    resultY = y;
                    success = true;
                    entropy = localEntropy;
                }
            }
        }

        return success;
    }

    private double GetEntropy(int x, int y)
    {
        return _probabilities.GetLastLengthSquared(_outputElements, x, y, 0);
    }

    private void UpdateDistribution(int x, int y)
    {
        if (_output.GetValue(x, y) != -1) return;
        
        var probabilities = _model.CalculateDistribution(x, y, _output, _input);

        probabilities.Print();
        
        if (probabilities.GetShape().GetSizeAt(0) != _outputElements)
            throw new Exception("Model returned invalid probability distribution");

        double length = 0.0;

        for (var i = 0; i < _outputElements; i++)
        {
            length += System.Math.Max(probabilities.GetValue(i), 0);
        }
                    
        for (var i = 0; i < _outputElements; i++)
        {
            _probabilities.SetValue(System.Math.Max(probabilities.GetValue(i) / length, 0), x, y, i);
        }
    }

    public Tensor<int> GetOutput()
    {
        return _output;
    }
    

}