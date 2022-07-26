using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;

public class DenseLayer: Layer
{

    private readonly Shape _shape;
    
    private Tensor<Variable>? _weights = null;

    private Tensor<Variable> _value;

    public DenseLayer(Shape shape) : base("dense_layer")
    {
        _shape = shape;
    }

    public DenseLayer(Shape shape, string id) : base(id)
    {
        _shape = shape;
    }

    public override Shape GetOutputShape()
    {
        return _shape;
    }

    public override void Build(IDataSource source)
    {
        _weights = new Tensor<Variable>(Shape.Of(source.GetOutputShape(), _shape));
        _weights.SetValue(Variable.Of(2), 0, 0);
        _weights.SetValue(Variable.Of(3), 1, 0);
        _weights.SetValue(Variable.Of(0.5), 2, 0);

        // TODO populate weights randomly
        
        _value = _weights.Mul(source.GetValue());
    }

    public override Tensor<Variable> GetValue()
    {
        return _value;
    }
}