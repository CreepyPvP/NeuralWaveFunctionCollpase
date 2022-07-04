using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning.Layers;

public class DenseLayer: Layer
{

    private readonly Shape _shape;
    
    private Tensor? _weights = null;


    public DenseLayer(Shape shape) : base("dense_layer")
    {
        _shape = shape;
    }

    public DenseLayer(Shape shape, string id) : base(id)
    {
        _shape = shape;
    }

    public override void SetInputShape(Shape input)
    {
        _weights = new Tensor(Shape.Of(input, _shape));
        
        // TODO populate weights randomly
    }

    public override Shape GetOutputShape()
    {
        return _shape;
    }
}