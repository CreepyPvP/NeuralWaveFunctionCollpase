using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public abstract class Layer : IDataSource
{

    private string _id;

    protected Layer(string id)
    {
        _id = id;
    }

    public abstract Shape GetOutputShape();

    public abstract void Build(IDataSource input);
    public abstract Tensor<Variable> GetValue();
    
}