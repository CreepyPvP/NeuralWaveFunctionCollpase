using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;
using Newtonsoft.Json.Linq;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public abstract class Layer : IDataSource
{

    private string _id;

    protected Layer(string id)
    {
        _id = id;
    }


    public String Id => _id;
    
    public abstract Shape GetOutputShape();

    public abstract void Build(IDataSource input, SeededRandom random, JToken node);
    public abstract Tensor<Variable> GetValue();
    
}