using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class Grid<T>
{


    private readonly T[] _outputElements;

    private readonly DataContainer<int> _output;

    // width x height x outputComplexity
    private readonly Tensor _probabilities;

    // width x height x inputLayerCount
    private readonly Tensor _input;

    private readonly int _width;
    private readonly int _height;

    private readonly IWaveFunctionModel _model;
    
    private readonly SeededRandom _random;
    

    public Grid(int width, int height, T[] outputElements, IWaveFunctionModel model, int seed): 
        this(width, height, outputElements, new Tensor(Shape.Of(width, height, 0)), model, seed) {}

    public Grid(int width, int height, T[] outputElements, Tensor input, IWaveFunctionModel model, int seed): 
        this(width, height, outputElements, new Tensor(Shape.Of(width, height, outputElements.Length)), input, model, seed) { }

    public Grid(int width, int height, T[] outputElements, Tensor initialStates, Tensor input, IWaveFunctionModel model, int seed)
    {
        _outputElements = outputElements;
        _probabilities = new Tensor(Shape.Of(width, height, outputElements.Length), 1.0);
        _input = input;

        _width = width;
        _height = height;

        _output = new DataContainer<int>(Shape.Of(width, height), -1);

        _model = model;
        
        if (initialStates.GetShape().GetSizeAt(0) != width || 
            initialStates.GetShape().GetSizeAt(1) != height ||
            initialStates.GetShape().GetSizeAt(2) != outputElements.Length ||
            initialStates.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid initial state");
        
        if (input.GetShape().GetSizeAt(0) != width || 
            input.GetShape().GetSizeAt(1) != height ||
            input.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid input format");

        _random = new SeededRandom(seed);
        
        // CalculateInitialDistribution();
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
        var x = 0;
        var y = 0;
        while (GetLowestEntropy(ref x, ref y)) 
        {
            var probabilities = _probabilities.Slice(2, x, y, 0);
            var collapsedElement = _random.NextIndex(probabilities, false);

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

    private bool GetLowestEntropy(ref int resultX, ref int resultY)
    {
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
        return _probabilities.GetLastLengthSquared(_outputElements.Length, x, y, 0);
    }

    private void UpdateDistribution(int x, int y)
    {
        if (_output.GetValue(x, y) != -1) return;
        
        var probabilities = _model.CalculateDistribution(x, y, _output, _input);

        if (probabilities.GetShape().GetSizeAt(0) != _outputElements.Length)
            throw new Exception("Model returned invalid probability distribution");

        double length = 0.0;

        for (var i = 0; i < _outputElements.Length; i++)
        {
            length += probabilities.GetValue(i);
        }
                    
        for (var i = 0; i < _outputElements.Length; i++)
        {
            _probabilities.SetValue(probabilities.GetValue(i) / length, x, y, i);
        }
    }

    public DataContainer<int> GetOutput()
    {
        return _output;
    }
    

}