using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class Grid
{


    private int _outputComplexity;

    private readonly DataContainer<int> _output;

    // width x height x outputComplexity
    private readonly Tensor _probabilities;

    // width x height x inputLayerCount
    private readonly Tensor _input;

    private readonly int _width;
    private readonly int _height;
    

    public Grid(int width, int height, int outputComplexity, Tensor input): 
        this(width, height, outputComplexity, new Tensor(Shape.Of(width, height, outputComplexity)), input) { }

    public Grid(int width, int height, int outputComplexity, Tensor initialStates, Tensor input)
    {
        _outputComplexity = outputComplexity;
        _probabilities = new Tensor(Shape.Of(width, height, outputComplexity), 1.0);
        _input = input;

        _width = width;
        _height = height;

        _output = new DataContainer<int>(Shape.Of(width, height), -1);
        
        if (initialStates.GetShape().GetSizeAt(0) != width || 
            initialStates.GetShape().GetSizeAt(1) != height ||
            initialStates.GetShape().GetSizeAt(2) != outputComplexity ||
            initialStates.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid initial state");
        
        if (input.GetShape().GetSizeAt(0) != width || 
            input.GetShape().GetSizeAt(1) != height ||
            input.GetShape().GetDimensionality() != 3)
            throw new Exception("Invalid input format");
    }


    public void Collapse()
    {
        int[]? currentPos;
        while ((currentPos = GetLowestEntropy()) != null)
        {

            
            
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
        for (var i = 0; i < _outputComplexity; i++)
        {
            var probability = _probabilities.GetValue(x, y, i);

            entropy += probability * probability;
        }

        return entropy;
    }
    
}