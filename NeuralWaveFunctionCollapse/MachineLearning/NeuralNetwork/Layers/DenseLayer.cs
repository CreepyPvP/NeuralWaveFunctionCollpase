using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;

public class DenseLayer: Layer
{

    private readonly Shape _shape;
    
    private Tensor<Variable>? _weights = null;
    private Tensor<Variable>? _biasWeights = null;

    private Func<Variable, Variable> _activation;

    private Tensor<Variable> _value;
    

    public DenseLayer(Shape shape, Func<Variable, Variable> activation, String id) : base(id)
    {
        _shape = shape;
        _activation = activation;
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

        _biasWeights = new Tensor<Variable>(_shape);
        _biasWeights.GetShape().ForEach(pos =>
        {
            _biasWeights.SetValue(Variable.Of(random.NextDouble()), pos);
        });
        
        _value = _weights.Mul(source.GetValue());
        _value.GetShape().ForEach(pos =>
        {
            _value.SetValue(_activation(_value.GetValue(pos) + _biasWeights.GetValue(pos)), pos);
        });
    }

    public override Tensor<Variable> GetValue()
    {
        return _value;
    }

    
    public override string ToString()
    {
        return $"{{id: \"{Id}\", weights: {_weights}, biases: {_biasWeights}}}";
    }
    
}