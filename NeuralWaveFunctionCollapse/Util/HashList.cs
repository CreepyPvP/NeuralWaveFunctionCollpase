namespace NeuralWaveFunctionCollapse.Util;


class ListElement<T>
{
    
    
    
}

// key => position, value => entropy
public class HashList<Key, Value>
{

    private readonly Dictionary<Key, int> _indexAccess = new();

    private readonly List<(Key, Value)> _values = new();


    public HashList()
    {
        // 1, 2, 3, 4, 5
        // 1, 2, --, 4, 5
        
        // 
    }


    public void Add(List<(Key, Value)> values)
    {
        
    }

    public (Key, Value)? Pop()
    {
        if (_values.Count == 0) return null;

        var entry = _values[0];

        _values.RemoveAt(0);

        _indexAccess.Remove(entry.Item1);
        
        return entry;
    }

    public void Update()
    {
        
    }

}