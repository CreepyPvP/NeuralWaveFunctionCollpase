using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;

public class DenseLayer: Layer
{

    private readonly Shape _shape;
    
    private Tensor? _weights = null;

    private IDataSource _input;
    private Tensor? _output = null;


    public DenseLayer(Shape shape) : base("dense_layer")
    {
        _shape = shape;
    }

    public DenseLayer(Shape shape, string id) : base(id)
    {
        _shape = shape;
    }

    public override void RegisterInput(IDataSource source)
    {
        _input = source;
        
        _weights = new Tensor(Shape.Of(source.GetOutputShape(), _shape));
        _weights.SetValue(5, 0, 0);
        _weights.SetValue(3, 1, 0);
        _weights.SetValue(0.5, 2, 0);

        // TODO populate weights randomly
    }

    public override Shape GetOutputShape()
    {
        return _shape;
    }

    public override Tensor GetData()
    {
        if (_weights == null) throw new Exception("Trying to compute with a non-compiled model");
        
        if (_output != null)
        {
            return _output;
        }

        _output = _weights.Mul(_input.GetData());
        return _output;
    }

    public override void Flush()
    {
        _output = null;
        
        _input.Flush();
    }
    
}