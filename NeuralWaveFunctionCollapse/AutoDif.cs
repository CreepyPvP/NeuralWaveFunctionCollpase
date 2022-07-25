using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse;

public class AutoDif
{

    public static void Start()
    {
        var x = Variable.Of(4);
        var y = Variable.Of(10);

        var results = F(x, y).Derive();
        
        Console.WriteLine(results[x]);
        Console.WriteLine(results[y]);
    }

    static Variable F(Variable x, Variable y)
    {
        return x * x + y * y;
    }
    
}