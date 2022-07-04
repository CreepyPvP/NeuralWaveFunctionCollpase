using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public abstract class Layer : IDataSource
{

    private string _id;

    protected Layer(string id)
    {
        _id = id;
    }

    public abstract void RegisterInput(IDataSource source);

    public abstract Shape GetOutputShape();

    public abstract Tensor GetData();
    public abstract void Flush();
    
}