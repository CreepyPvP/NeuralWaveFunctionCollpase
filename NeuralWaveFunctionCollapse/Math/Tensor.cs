using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.Math;


public class Shape
{

    private readonly int[] _dimensions;

    private Shape(int[] dimensions)
    {
        _dimensions = dimensions;
    }

    public int Size(int dimension = 0)
    {
        int size = 1;

        for (int i = dimension; i < _dimensions.Length; i++)
        {
            size *= _dimensions[i];
        }

        return size;
    }

    /*
     *  3x2
     *  0 2 4
     *  1 3 5
     */
    public int GetIndex(params int[] position)
    {
        if (position.Length != _dimensions.Length)
            throw new Exception("Non matching dimensionality: " + position.Length + " != " + _dimensions.Length);

        var index = 0;
        for (var i = 0; i < position.Length; i++)
        {
            index += position[i] * Size(i + 1);
        }

        return index;
    }

    public int GetDimensionality()
    {
        return _dimensions.Length;
    }

    public int GetLength(int dimension)
    {
        return _dimensions[dimension];
    }

    public void ForEach(Action<int[]> action)
    {
        var pos = new int[_dimensions.Length];
        ForEachRecursive(action, pos, 0);
    }

    private void ForEachRecursive(Action<int[]> action, int[] currentPos, int dimension)
    {
        if (dimension >= _dimensions.Length)
        {
            action.Invoke(currentPos);
            return;
        }

        for (var pos = 0; pos < _dimensions[dimension]; pos++)
        {
            currentPos[dimension] = pos;
            ForEachRecursive(action, currentPos, dimension + 1);
        }
    }


    public int GetSizeAt(int dimension)
    {
        return _dimensions[dimension];
    }

    public override bool Equals(object? obj)
    {
        if (obj is not Shape) return false;

        var shape = (Shape) obj;
        
        if (_dimensions.Length != shape._dimensions.Length) return false;

        return !_dimensions.Where((t, i) => t != shape._dimensions[i]).Any();
    }

    public static Shape Of(params int[] shape)
    {
        return new Shape(shape);
    }

    public static Shape Of(Shape s0, Shape s1)
    {
        var result = s0._dimensions.ArrJoin(s1._dimensions);

        return new Shape(result);
    }

    public static Shape Sub(Shape shape, int dimension)
    {
        var dimensions = new int[shape._dimensions.Length - dimension];

        for (int i = 0; i < dimensions.Length; i++)
        {
            dimensions[i] = shape._dimensions[i + dimension];
        }
        
        return new Shape(dimensions);
    }
    
    
}

public class DataContainer<T>
{

    private readonly Shape _shape;

    private readonly T[] _values;
    
    public DataContainer(Shape shape)
    {
        _shape = shape;
        _values = new T[_shape.Size()];
    }

    public DataContainer(Shape shape, T initial)
    {
        _shape = shape;
        var size = _shape.Size();
        _values = new T[size];

        for (var i = 0; i < size; i++)
        {
            _values[i] = initial;
        }
    }


    public T GetValue(params int[] position)
    {
        return _values[_shape.GetIndex(position)];
    }

    public void SetValue(T value, params int[] position)
    {
        _values[_shape.GetIndex(position)] = value;
    }

    public Shape GetShape()
    {
        return _shape;
    }


    public DataContainer<T> Slice(int dimension, params int[] position)
    {
        // TODO: checks

        var posCopy = position.Copy();
        
        var outputShape = Shape.Of(_shape.GetSizeAt(dimension) - position[dimension]);
        var output = new DataContainer<T>(outputShape);

        for (var i = 0; i < outputShape.GetSizeAt(0); i++)
        {
            output.SetValue(GetValue(posCopy));
            posCopy[dimension]++;
        }
        
        return output;
    }

}

public class Tensor : DataContainer<double>
{
    public Tensor(Shape shape) : base(shape)
    {
    }
    
    public Tensor(Shape shape, double initialValue) : base(shape, initialValue)
    {
    }
    
}