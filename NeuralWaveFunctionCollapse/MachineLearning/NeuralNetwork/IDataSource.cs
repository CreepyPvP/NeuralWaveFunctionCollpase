using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public interface IDataSource
{

    Shape GetOutputShape();
    
    void Build(IDataSource source);

    Tensor<Variable> GetValue();

}


public class InputDataSource : IDataSource
{

    private readonly Shape _shape;
    private readonly Tensor<Variable> _output;

    public InputDataSource(Shape shape)
    {
        _shape = shape;
        _output = new Tensor<Variable>(_shape);

        var rawOutput = _output.GetRaw();
        for (var i = 0; i < _shape.Size(); i++)
        {
            rawOutput[i] = Variable.Of(0, false);
        }
    }
    
    
    public Shape GetOutputShape()
    {
        return _shape;
    }

    public void Build(IDataSource source)
    {

    }

    public Tensor<Variable> GetValue()
    {
        return _output;
    }

    public void SetInput(Tensor<double> tensor, bool disableChecks = false)
    {
        if (!disableChecks && !_shape.Equals(tensor.GetShape()))
            throw new Exception("Shape of input tensor does not match required shape");
        
        tensor.GetShape().ForEach(pos =>
        {
           _output.GetValue(pos).Set(tensor.GetValue(pos));
        });
    }
    
}