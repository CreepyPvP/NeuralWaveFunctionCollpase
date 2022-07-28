using System.Diagnostics;
using System.Threading.Channels;

namespace NeuralWaveFunctionCollapse.Math.AutoDif;

public class Variable
{

    private IOperation _source;

    private readonly bool _derivable;


    private Variable(IOperation value, bool derivable = true)
    {
        _source = value;
        _derivable = derivable;
    }


    public double Value()
    {
        return _source.GetValue();
    }
    
    public void Values(Dictionary<Variable, double> values, Dictionary<Variable, List<Variable>> dependants)
    {
        foreach (var dependency in _source.GetDependencies())
        {
            if (!dependants.ContainsKey(dependency))
                dependants[dependency] = new List<Variable>();

            dependants[dependency].Add(this);
        }

        values[this] = _source.GetValues(values, dependants);
    }
    
    public void Values(Dictionary<Variable, double> values)
    {
        values[this] = _source.GetValues(values);
    }
    

    public Dictionary<Variable, double> Derive()
    {
        return Derive(new Dictionary<Variable, double>());  
    }
    
    public Dictionary<Variable, double> Derive(Dictionary<Variable, double> values)
    {
        Dictionary<Variable, double> derivatives = new();
        Dictionary<Variable, List<Variable>> dependants = new();

        Values(values, dependants);
        DeriveAll(derivatives, values, dependants, true);
        
        return derivatives;
    }

    private void DeriveAll(Dictionary<Variable, double> derivatives,
        Dictionary<Variable, double> values,
        Dictionary<Variable, List<Variable>> dependants,
        bool isRoot = false)
    {
        if (!_derivable) return;

        var derivative = 0.0;
        
        if (!isRoot)
        {
            if (dependants.ContainsKey(this))
            {
                var dep = dependants[this];

                foreach (var dependant in dep)
                {
                    if (!derivatives.ContainsKey(dependant))
                        return;
                    
                    derivative += derivatives[dependant] * dependant.Derive(this, values);
                }
            }
    
        }
        else
        {
            derivative = 1.0;
        }

        derivatives[this] = derivative;
        
        foreach (var dependency in _source.GetDependencies())
        {
            dependency.DeriveAll(derivatives, values, dependants);
        }
    }


    private double Derive(Variable to, Dictionary<Variable, double> values)
    {
        return _source.Derive(to, values);
    }

    public void Set(double value)
    {
        _source = new Identity(value);
    }


    public bool IsIdentity()
    {
        return _source is Identity;
    }
    
    // Operators ------------------------------------------------------------------
    
    public static Variable operator *(Variable var1, Variable var2)
    {
        return Mul(var1, var2);
    }
    
    public static Variable operator *(Variable var1, double var2)
    {
        return Mul(var1, Variable.Of(var2, false));
    }
    
    public static Variable operator *(double var1, Variable var2)
    {
        return Mul(var2, Variable.Of(var1, false));
    }
    
    public static Variable operator +(Variable var1, Variable var2)
    {
        return Add(var1, var2);
    }
    
    public static Variable operator +(Variable var1, double var2)
    {
        return Add(var1, Variable.Of(var2, false));
    }
    
    public static Variable operator +(double var1, Variable var2)
    {
        return Add(Variable.Of(var1, false), var2);
    }
    
    public static Variable operator -(Variable var1)
    {
        return Mul(var1, Variable.Of(-1, false));
    }
    
    public static Variable operator -(Variable var1, Variable var2)
    {
        return Add(var1, -var2);
    }
    
    public static Variable operator -(Variable var1, double var2)
    {
        return Add(var1, Variable.Of(-var2, false));
    }
    
    public static Variable operator -(double var1, Variable var2)
    {
        return Add(Variable.Of(var1, false), -var2);
    }
    
    public static Variable operator /(Variable var1, Variable var2)
    {
        return Mul(var1, Invert(var2));
    }
    
    public static Variable operator /(Variable var1, double var2)
    {
        return Mul(var1, Invert(Variable.Of(var2, false)));
    }
    
    public static Variable operator /(double var1, Variable var2)
    {
        return Mul(Variable.Of(var1), Invert(var2));
    }

    public static Variable Of(double value, bool derivable = true)
    {
        return new Variable(new Identity(value), derivable);
    }
    
    private static Variable Mul(Variable var1, Variable var2)
    {
        return new Variable(new Multiply(var1, var2));
    }


    public static Variable Add(Variable var1, Variable var2)
    {
        return new Variable(new Add(var1, var2));
    }


    public static Variable Invert(Variable var1)
    {
        return new Variable(new Invert(var1));
    }


    public static Variable Max(Variable var1, double var2)
    {
        return new Variable(new Max(var1, var2));
    }

}