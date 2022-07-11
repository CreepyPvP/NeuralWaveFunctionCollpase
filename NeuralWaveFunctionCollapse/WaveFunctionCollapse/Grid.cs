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

    private readonly SeededRandom _random;
    

    public Grid(int width, int height, T[] outputElements, Tensor input, int seed): 
        this(width, height, outputElements, new Tensor(Shape.Of(width, height, outputElements.Length)), input, seed) { }

    public Grid(int width, int height, T[] outputElements, Tensor initialStates, Tensor input, int seed)
    {
        _outputElements = outputElements;
        _probabilities = new Tensor(Shape.Of(width, height, outputElements.Length), 1.0);
        _input = input;

        _width = width;
        _height = height;

        _output = new DataContainer<int>(Shape.Of(width, height), -1);
        
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
    }


    public void Collapse()
    {
        int[]? currentPos;
        while ((currentPos = GetLowestEntropy()) != null)
        {
            var probabilities = _probabilities.Slice(2, 0, 0, 0);
            var collapsedElement = _random.NextIndex(probabilities, false);
            
            _output.SetValue(collapsedElement, currentPos);
            
            // TODO propagate changes
            
        }
    }


    private int[]? GetLowestEntropy()
    {
        int[]? pos = null;
        double entropy = 2;
        
        for (var x = 0; x < _width; x++)
        {
            for (var y = 0; y < _height; y++)
            {
                var localEntropy = GetEntropy(x, y);
                if (_output.GetValue(x, y) == -1 && localEntropy < entropy)
                {
                    pos = new[] { x, y };
                    entropy = localEntropy;
                }
            }
        }

        return pos;
    }

    private double GetEntropy(int x, int y)
    {
        double entropy = 0;
        for (var i = 0; i < _outputElements.Length; i++)
        {
            var probability = _probabilities.GetValue(x, y, i);

            entropy += probability * probability;
        }

        return entropy;
    }
    
}