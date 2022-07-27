using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;

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

    public override void Build(IDataSource source, SeededRandom random)
    {
        _weights = new Tensor<Variable>(Shape.Of(source.GetOutputShape(), _shape));

        _weights.GetShape().ForEach(pos =>
        {
            _weights.SetValue(Variable.Of((random.NextDouble() * 2) - 1), pos);
        });
        
        _value = _weights.Mul(source.GetValue());
    }

    public override Tensor<Variable> GetValue()
    {
        return _value;
    }
}