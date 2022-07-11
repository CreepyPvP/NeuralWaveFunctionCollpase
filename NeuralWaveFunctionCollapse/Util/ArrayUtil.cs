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

    public static int[] CopyAndAdd(this int[] self, int index, int steps)
    {
        var copy = new int[self.Length];

        for (var i = 0; i < copy.Length; i++)
        {
            copy[i] = index == i ? self[i] + steps : self[i];
        }

        return copy;
    }


    public static T[] Copy<T>(this T[] self)
    {
        var copy = new T[self.Length];
        
        for (var i = 0; i < copy.Length; i++)
        {
            copy[i] = self[i];
        }

        return copy;
    }
    
}