using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.Types;

public class LdtkLevel
{

    public string identifier;

    public LdtkLayerInstance[] layerInstances;
    public LdtkField[] fieldInstances;



    public LdtkField GetField(string name)
    {
        return fieldInstances.Single(field => field.__identifier.Equals(name));
    }
        
    public LdtkLayerInstance GetLayer(string name)
    {
        return layerInstances.Single(field => field.__identifier.Equals(name));
    }

}


public class LdtkLayerInstance
{
        
    public string __identifier;

    public int __cWid;
    public int __cHei;

    public int[] intGridCsv;


    public LdtkLayerInstance()
    {
        
    }

    public LdtkLayerInstance(Tensor<int> from, string identifier)
    {
        if (from.GetShape().GetDimensionality() != 2)
            throw new Exception("Invalid data container");

        __identifier = identifier;
        __cWid = from.GetShape().GetSizeAt(0);
        __cHei = from.GetShape().GetSizeAt(1);

        intGridCsv = from.GetRaw();
    }
    

    public int GetTile(int x, int y)
    {
        return intGridCsv[x + y * __cWid];
    }


    public Tensor<double> ToTensor()
    {
        var result = new Tensor<double>(Shape.Of(__cWid, __cHei));

        for (var x = 0; x < __cWid; x++)
        {
            for (var y = 0; y < __cHei; y++)
            {
                result.SetValue(intGridCsv[x + __cWid * y], x, y);
            }
        }

        return result;
    }
    

}

public class LdtkField
{
    public string __identifier;
    public string __value;

}