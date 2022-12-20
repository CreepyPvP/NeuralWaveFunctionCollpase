using NeuralWaveFunctionCollapse.Math.AutoDif;
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

        for (var i = dimension; i < _dimensions.Length; i++)
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

    public static Shape Of(Shape shape, int from, int count)
    {
        return new Shape(new ArraySegment<int>(shape._dimensions, from, count).ToArray());
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

public class Tensor<T>
{

    protected readonly Shape _shape;

    protected readonly T[] _values;
    
    public Tensor(Shape shape)
    {
        _shape = shape;
        _values = new T[_shape.Size()];
    }

    public Tensor(Tensor<T> copy)
    {
        _shape = copy._shape;
        _values = copy._values;
    }

    public Tensor(Shape shape, T[] initialValues)
    {
        if (initialValues.Length != shape.Size())
            throw new Exception("Invalid initial values size");

        _shape = shape;
        _values = initialValues;
    }

    public Tensor(Shape shape, T initial)
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


    public Tensor<T> Slice(int dimension, params int[] position)
    {
        // TODO: checks

        var posCopy = position.Copy();
        
        var outputShape = Shape.Of(_shape.GetSizeAt(dimension) - position[dimension]);
        var output = new Tensor<T>(outputShape);

        for (var i = 0; i < outputShape.GetSizeAt(0); i++)
        {
            output.SetValue(GetValue(posCopy), i);
            posCopy[dimension]++;
        }
        
        return output;
    }


    public Tensor<T>[] ToArray()
    {
        var shape = _shape.GetDimensionality() == 1 ? Shape.Of(1) : Shape.Of(_shape, 1, _shape.GetDimensionality() - 1);

        var position = new int[_shape.GetDimensionality()];
        
        var result = new Tensor<T>[_shape.GetSizeAt(0)];
        for (var i = 0; i < _shape.GetSizeAt(0); i++)
        {
            position[0] = i;
            
            var values = new ArraySegment<T>(_values,_shape.GetIndex(position), _shape.Size(1)).ToArray();
            result[i] = new Tensor<T>(shape, values);
        }

        return result;
    }

    public Tensor<T> ByIndexContainer(Tensor<int> indexContainer)
    {
        var result = new Tensor<T>(indexContainer.GetShape());

        for (var i = 0; i < indexContainer._values.Length; i++)
        {
            result._values[i] = _values[indexContainer._values[i]];
        }
        
        return result;
    }

    public void Copy(Tensor<T> toCopy, int index = 0)
    {
        toCopy._values.CopyTo(_values, index);
    }

    public T[] GetRaw()
    {
        return _values;
    }
    
    public void Print(bool jsonArray = false)
    {
        if (this._shape.GetDimensionality() == 2)
        {
            Print2D(jsonArray);
            return;
        }

        if (this._shape.GetDimensionality() == 1)
        {
            Print1D(jsonArray);
            return;
        }

        throw new Exception("Print not implemented for this dimensionality");
    }

    private void Print1D(bool jsonArray)
    {
        var output = jsonArray ? "[" : "";
        
        for (var x = 0; x < _shape.GetSizeAt(0); x++)
        {
            output += ((x == 0) ? GetValue(x) : ", " + GetValue(x));
        }

        if (jsonArray) output += "]";
        
        Console.WriteLine(output);
    }

    private void Print2D(bool jsonArray)
    {
        var output = jsonArray ? "[" : "";
        
        for (var y = 0; y < _shape.GetSizeAt(1); y++)
        {
            for (var x = 0; x < _shape.GetSizeAt(0); x++)
            {
                output += ((x == 0) ? GetValue(x, y) : ", " + GetValue(x, y));
            }

            output += jsonArray? ",\n" : "\n";
        }

        if (jsonArray) output += "]";

        Console.WriteLine(output);
    }
    
    public Tensor<T> UpDimension()
    {
        var resultShape = Shape.Of(_shape, Shape.Of(1));
        return new Tensor<T>(resultShape, _values);
    }
    
    
    
    
    
    public Tensor<TR> Compare<T2, TR>(Tensor<T2> arg1, Func<T, T2, TR> func)
    {
        var result = new Tensor<TR>(_shape);

        for (var i = 0; i < _values.Length; i++)
        {
            result._values[i] = func(_values[i], arg1._values[i]);
        }

        return result;
    }

    public void ForEach(Action<T> action)
    {
        foreach (var value in _values)
        {
            action(value);
        }
    }


    public override string ToString()
    {
        var content = String.Join(",", _values.Select(value => "\"" + value.ToString() + "\""));
        return $"[{content}]";
    }
}