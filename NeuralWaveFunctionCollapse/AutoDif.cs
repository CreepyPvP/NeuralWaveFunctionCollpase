using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse;

public class AutoDif
{

    public static void Start()
    {
        var v1 = Variable.Of(4);
        var v2 = Variable.Of(10);

        var derivatives = F(v1, v2).Derive();
        
        Console.WriteLine(derivatives[v1]);
        Console.WriteLine(derivatives[v2]);
    }


    static Variable F(Variable var1, Variable var2)
    {
        return var1 * var2;
    }
    
}