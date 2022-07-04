using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public abstract class Layer
{

    private string _id;

    protected Layer(string id)
    {
        _id = id;
    }

    public abstract void SetInputShape(Shape input);

    public abstract Shape GetOutputShape();


    public void SetId(string id)
    {
        _id = id;
    }

    public string GetId()
    {
        return _id;
    }

}