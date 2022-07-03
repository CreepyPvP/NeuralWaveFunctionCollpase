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
        
        for(int i = dimension; i < _dimensions.Length; i++)
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
    
    public static Shape Of(params int[] shape)
    {
        return new Shape(shape);
    }
    
}

public class Matrix
{

    private readonly Shape _shape;

    private readonly double[] _values;
    
    public Matrix(Shape shape)
    {
        _shape = shape;
        _values = new double[_shape.Size()];
    }

    public double GetValue(params int[] position)
    {
        return _values[_shape.GetIndex(position)];
    }

    public double SetValue(double value, params int[] position)
    {
        _values[_shape.GetIndex(position)] = value;
    }
    
}