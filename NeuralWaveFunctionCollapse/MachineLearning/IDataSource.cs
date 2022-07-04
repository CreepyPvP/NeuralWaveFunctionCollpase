using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public interface IDataSource
{

    Shape GetOutputShape();
    
    Tensor GetData();

    void Flush();

}


public class InputDataSource : IDataSource
{

    private readonly Shape _shape;
    private Tensor? _output = null;

    public InputDataSource(Shape shape)
    {
        _shape = shape;
    }
    
    
    public Shape GetOutputShape()
    {
        return _shape;
    }

    public Tensor GetData()
    {
        if (_output == null) throw new Exception("No input data was provided. This should not happen");
        
        return _output;
    }

    public void Flush()
    {
        _output = null;
    }

    public void SetInput(Tensor tensor)
    {
        // TODO check for compatibility with _shape
        
        _output = tensor;
    }
    
}