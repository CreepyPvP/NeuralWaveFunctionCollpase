using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.Util;




public class SeededRandom
{

    private Random _random;

    public SeededRandom(int seed)
    {
        _random = new Random(seed);
    }

    public int NextInt()
    {
        return _random.Next();
    }

    public T NextElement<T>(T[] elements)
    {
        var index = _random.Next(elements.Length);
        return elements[index];
    }

    /**
     *  Selects a random element from elements, which is NOT part of usedElements
     */
    public T NextElement<T>(T[] elements, List<T> usedElements)
    {
        var index = _random.Next(elements.Length - usedElements.Count);
        for (int i = 0; i < elements.Length; i++)
        {
            if (usedElements.Contains(elements[i])) index++;

            if (index == i) return elements[index];
        }

        throw new Exception("This happens, when you pass more usedElements then elements. Let this be you a lesson...");
    }

    public int NextIndex(DataContainer<double> distribution, bool distributionNormed, bool ignoreChecks = false)
    {
        var elements = distribution.GetShape().GetSizeAt(0);
        
        if (!ignoreChecks && (
                distribution.GetShape().GetDimensionality() != 1))
            throw new Exception("Invalid distribution");

        double totalLength = distributionNormed? 1 : 0;
        if (!distributionNormed)
        {
            for (var i = 0; i < elements; i++)
            {
                totalLength += distribution.GetValue(i);
            }
        }

        var number = _random.NextDouble() * totalLength;
        double currentSum = 0;
        
        // distribution.Print();

        for (var i = 0; i < elements; i++)
        {
            currentSum += distribution.GetValue(i);
            if (number <= currentSum) return i;
        }

        throw new Exception("This should not happen");
    }
    
    public double NextDouble(double mean, double stdDev)
    {
        double u1 = 1.0-_random.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0-_random.NextDouble();
        double randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) *
                               System.Math.Sin(2.0 * System.Math.PI * u2); //random normal(0,1)
        return mean + stdDev * randStdNormal;
    }

    // Totally not copied from SO
    public void Shuffle<T>(T[] arr)
    {
        int n = arr.Length;
        while (n > 1) 
        {
            var k = _random.Next(n--);
            (arr[n], arr[k]) = (arr[k], arr[n]);
        }
    }
    
}

