using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse;

public class AutoDif
{

    public static void Start()
    {
        var v1 = Variable.Of(4);
        var v2 = Variable.Of(10);

        var optimiser = new StochasticGradientDescentOptimiser();

        var config = new SgdConfig
        {
            Iterations = 10000,
            LearnRate = 0.001
        };

        optimiser.Minimize(F(v1, v2), new Variable[]{ v1, v2 }, config);
        
        Console.WriteLine(v1.Value());
        Console.WriteLine(v2.Value());
    }
    

    static Variable F(Variable x, Variable y)
    {
        var var1 = x - 5;
        var var2 = y + 10;
        return var1 * var1 + var2 * var2;
    }
    
}