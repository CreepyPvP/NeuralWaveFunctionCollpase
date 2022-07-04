namespace NeuralWaveFunctionCollapse.Math;

public class GraphNode<T>
{

    private readonly T _value;
    private readonly List<GraphNode<T>> _children = new List<GraphNode<T>>();

    public GraphNode(T value)
    {
        _value = value;
    }

    public void Add(GraphNode<T> child)
    {
        _children.Add(child);
    }

    public void ForEach(Action<GraphNode<T>> action)
    {
        action.Invoke(this);
        
        _children.ForEach(child => child.ForEach(action));
    }

    public T GetValue()
    {
        return _value;
    }

    public List<GraphNode<T>> GetChildren()
    {
        return _children;
    }

}