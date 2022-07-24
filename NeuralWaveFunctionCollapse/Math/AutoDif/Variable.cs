using System.Threading.Channels;

namespace NeuralWaveFunctionCollapse.Math.AutoDif;

public class Variable
{

    private readonly IOperation _source;
    
    private readonly List<Variable> _dependants = new();



    private Variable(IOperation value)
    {
        _source = value;
    }


    public double Value()
    {
        return _source.GetValue();
    }
    
    public void Values(Dictionary<Variable, double> values)
    {
        values[this] = _source.GetValues(values);
    }
    

    public Dictionary<Variable, double> Derive()
    {
        Dictionary<Variable, double> derivatives = new();
        Dictionary<Variable, double> values = new();

        Values(values);
        DeriveAll(derivatives, values, 1);
        
        return derivatives;
    }

    private void DeriveAll(Dictionary<Variable, double> derivatives, Dictionary<Variable, double> values, double? derivative = null)
    {
        if (!derivative.HasValue)
        {
            derivative = 0.0;
            foreach (var dependant in _dependants)
            {
                // derivative not present; stop; will get called later by dependant once its derivative has been calculated
                if (!derivatives.ContainsKey(dependant)) return;

                derivative += derivatives[dependant] * dependant.Derive(this, values);
            }
        }

        derivatives[this] = derivative!.Value;
        
        foreach (var dependency in _source.GetDependencies())
        {
            dependency.DeriveAll(derivatives, values);
        }
    }


    private double Derive(Variable to, Dictionary<Variable, double> values)
    {
        return _source.Derive(to, values);
    }
    
    
    // Operators ------------------------------------------------------------------
    
    public static Variable operator *(Variable var1, Variable var2)
    {
        return Mul(var1, var2);
    }
    
    public static Variable operator *(Variable var1, double var2)
    {
        return Mul(var1, Variable.Of(var2));
    }
    
    public static Variable operator *(double var1, Variable var2)
    {
        return Mul(var2, Variable.Of(var1));
    }
    
    public static Variable operator +(Variable var1, Variable var2)
    {
        return Add(var1, var2);
    }
    
    public static Variable operator +(Variable var1, double var2)
    {
        return Add(var1, Variable.Of(var2));
    }
    
    public static Variable operator +(double var1, Variable var2)
    {
        return Add(Variable.Of(var1), var2);
    }
    
    public static Variable operator -(Variable var1)
    {
        return Mul(var1, Variable.Of(-1));
    }
    
    public static Variable operator -(Variable var1, Variable var2)
    {
        return Add(var1, -var2);
    }
    
    public static Variable operator -(Variable var1, double var2)
    {
        return Add(var1, Variable.Of(-var2));
    }
    
    public static Variable operator -(double var1, Variable var2)
    {
        return Add(Variable.Of(var1), -var2);
    }
    
    public static Variable operator /(Variable var1, Variable var2)
    {
        return Mul(var1, Invert(var2));
    }
    
    public static Variable operator /(Variable var1, double var2)
    {
        return Mul(var1, Invert(Variable.Of(var2)));
    }
    
    public static Variable operator /(double var1, Variable var2)
    {
        return Mul(Variable.Of(var1), Invert(var2));
    }

    public static Variable Of(double value)
    {
        return new Variable(new Identity(value));
    }
    
    private static Variable Mul(Variable var1, Variable var2)
    {
        var result = new Variable(new Multiply(var1, var2));
        var1._dependants.Add(result);
        var2._dependants.Add(result);
        return result;
    }


    public static Variable Add(Variable var1, Variable var2)
    {
        var result = new Variable(new Add(var1, var2));
        var1._dependants.Add(result);
        var2._dependants.Add(result);
        return result;
    }


    public static Variable Invert(Variable var1)
    {
        var result = new Variable(new Invert(var1));
        var1._dependants.Add(result);
        return result;
    }

}