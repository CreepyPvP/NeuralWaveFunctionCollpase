namespace NeuralWaveFunctionCollapse.Util;

public static class ArrayUtil
{

    public static T[] ArrJoin<T>(this T[] arr0, T[] arr1)
    {
        var z = new T[arr0.Length + arr1.Length];
        arr0.CopyTo(z, 0);
        arr1.CopyTo(z, arr0.Length);

        return z;
    }
    
}