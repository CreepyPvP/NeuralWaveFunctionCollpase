using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse;

public class AutoDif
{

    public static void Start()
    {
        var v1 = Variable.Of(1);

        var benchmark = new Benchmark.Benchmark(() =>
        {
            var test = ExpensiveFunction(v1).Derive();
        }, 1);
        benchmark.Run();
        
        Console.WriteLine(benchmark.AvgTime);
    }


    static Variable ExpensiveFunction(Variable v)
    {
        for (var i = 0; i < 10000; i++)
        {
            v *= 10;
        }

        return v;
    }

    static Variable F(Variable x, Variable y)
    {
        var var1 = x - 5;
        var var2 = y + 10;
        return var1 * var1 + var2 * var2;
    }
    
}