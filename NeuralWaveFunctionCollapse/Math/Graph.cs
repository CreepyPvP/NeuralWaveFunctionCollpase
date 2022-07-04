namespace NeuralWaveFunctionCollapse.Math;

public class GraphNode<T>
{

    private readonly T _value;
    private readonly List<GraphNode<T>> _children;

    public GraphNode(T value)
    {
        _value = value;
    }

    public void Add(GraphNode<T> child)
    {
        _children.Add(child);
    }

}