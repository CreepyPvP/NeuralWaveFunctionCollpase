using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class Grid
{


    private int _outputComplexity;
    
    // width x height x outputComplexity
    private readonly Tensor _output;

    // width x height x inputLayerCount
    private readonly Tensor _input;
    

    public Grid(int width, int height, int outputComplexity, Tensor input): 
        this(width, height, outputComplexity, new Tensor(Shape.Of(width, height, outputComplexity)), input) { }

    public Grid(int width, int height, int outputComplexity, Tensor initialStates, Tensor input)
    {
        _outputComplexity = outputComplexity;
        _output = new Tensor(Shape.Of(width, height, outputComplexity));
        _input = input;

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
        
    }
    
}