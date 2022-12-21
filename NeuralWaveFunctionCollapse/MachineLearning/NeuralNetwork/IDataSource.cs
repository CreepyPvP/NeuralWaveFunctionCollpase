using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;
using Newtonsoft.Json.Linq;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public interface IDataSource
{

    Shape GetOutputShape();
    
    void Build(IDataSource source, SeededRandom random, JToken token);

    Tensor<Variable> GetValue();

}


public class InputDataSource : IDataSource
{

    private readonly Tensor<Variable> _output;

    public InputDataSource(Shape shape)
    {
        _output = new Tensor<Variable>(shape);

        var rawOutput = _output.GetRaw();
        for (var i = 0; i < shape.Size(); i++)
        {
            rawOutput[i] = Variable.Of(0, false);
        }
    }
    
    
    public Shape GetOutputShape()
    {
        return _output.GetShape();
    }

    public void Build(IDataSource source, SeededRandom random, JToken token)
    {

    }

    public Tensor<Variable> GetValue()
    {
        return _output;
    }

    public void SetInput(Tensor<double> tensor, bool disableChecks = false)
    {
        if (!disableChecks)
        {
            if(!_output.GetShape().Equals(tensor.GetShape()))
                throw new Exception("Shape of input tensor does not match required shape");   
        }

        tensor.GetShape().ForEach(pos =>
        {
           _output.GetValue(pos).Set(tensor.GetValue(pos));
        });
    }
    
}