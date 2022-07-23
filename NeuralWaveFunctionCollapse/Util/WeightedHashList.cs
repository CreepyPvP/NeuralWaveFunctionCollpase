namespace NeuralWaveFunctionCollapse.Util;



// key => position, value => entropy
public class WeightedHashList<TKey, TValue> where TValue: IWeighable where TKey : notnull
{

    private readonly Dictionary<TKey, LinkedListNode<(TKey, TValue)>> _indexAccess = new();

    private readonly LinkedList<(TKey, TValue)> _values = new();

    public void Add(List<(TKey, TValue)> values)
    {
        var first = _values.First;
        
        // add all as empty
        if (first == null)
        {
            InsertAllAt(values);
            return;
        }
        values.Sort(ReverseCompare);
        var next = first.Next;

        while (next != null)
        {
            for (var i = values.Count - 1; i >= 0; i--)
            {
                var item = values[i];
                var result = Compare(next.Value, item);
            
                // If larger than skip because the following values will always be bigger
                if (result > 0) break;
            
                var thisNode = new LinkedListNode<(TKey, TValue)>(item);
                _indexAccess.Add(item.Item1, thisNode);
                _values.AddAfter(next, thisNode);
                values.RemoveAt(i);
            }

            first = next;
            next = first.Next;
        }
        if(values.Count == 0) return;
        values.Sort(Compare);
        InsertAllAt(values, first);
    }

    private void InsertAllAt(List<(TKey, TValue)> values, LinkedListNode<(TKey, TValue)>? initParent = null)
    {
        values.Sort(Compare);
        var firstItem = values.First();
        var parent = new LinkedListNode<(TKey, TValue)>(values.First());
        _indexAccess.Add(firstItem.Item1, parent);
        if(initParent == null)
        {
            _values.AddFirst(parent);
        }
        else
        {
            _values.AddAfter(initParent, parent);
        }
        foreach (var item in _values.Skip(1))
        {
            var thisNode = new LinkedListNode<(TKey, TValue)>(values.First());
            _indexAccess.Add(item.Item1, thisNode);
            _values.AddAfter(parent, thisNode);
            parent = thisNode;
        }

        return;
    }

    private static int ReverseCompare((TKey, TValue) x, (TKey, TValue) y)
    {
        return -Compare(x, y);
    }
    private static int Compare((TKey, TValue) x, (TKey, TValue) y)
    {
        return (x.Item2.Weight - y.Item2.Weight) switch
        {
            > 0 => 1,
            < 0 => -1,
            _ => 0
        };
    }

    public (TKey, TValue)? Pop()
    {
        if (_values.Count == 0) return null;

        var entry = _values.First!.Value;

        _values.RemoveFirst();

        _indexAccess.Remove(entry.Item1);
        
        return entry;
    }

    /**
     * Requires all keys to be added!
     */
    public void Remove(IEnumerable<TKey> keys)
    {
        foreach (var key in keys)
        {
            var value = _indexAccess[key];
            _values.Remove(value);
            _indexAccess.Remove(key);
        }
    }

    public TValue? Get(TKey key)
    {
        return _indexAccess.TryGetValue(key, out var value) ? value.Value.Item2 : default;
    }
}