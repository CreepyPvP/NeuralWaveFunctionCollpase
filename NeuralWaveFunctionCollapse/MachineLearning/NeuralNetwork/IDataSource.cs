using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public interface IDataSource
{

    Shape GetOutputShape();
    
    Tensor<double> GetData();

    void Flush();

}


public class InputDataSource : IDataSource
{

    private readonly Shape _shape;
    private Tensor<double>? _output = null;

    public InputDataSource(Shape shape)
    {
        _shape = shape;
    }
    
    
    public Shape GetOutputShape()
    {
        return _shape;
    }

    public Tensor<double> GetData()
    {
        if (_output == null) throw new Exception("No input data was provided. This should not happen");
        
        return _output;
    }

    public void Flush()
    {
        _output = null;
    }

    public void SetInput(Tensor<double> tensor, bool disableChecks = false)
    {
        if (!disableChecks && !_shape.Equals(tensor.GetShape()))
            throw new Exception("Shape of input tensor does not match required shape");
        
        _output = tensor;
    }
    
}