namespace NeuralWaveFunctionCollapse.Math.AutoDif;

public interface IOperation
{

    double GetValue();

    double GetValues(Dictionary<Variable, double> valueStore);

    Variable[] GetDependencies();

    double Derive(Variable var, Dictionary<Variable, double> values);

}


public class Identity : IOperation
{

    private readonly double _value;
    
    public Identity(double value)
    {
        _value = value;
    }

    public double GetValue()
    {
        return _value;
    }

    public double GetValues(Dictionary<Variable, double> valueStore)
    {
        return _value;
    }

    public Variable[] GetDependencies()
    {
        return Array.Empty<Variable>();
    }

    public double Derive(Variable var, Dictionary<Variable, double> values)
    {
        return 0;
    }
}

public class Multiply: IOperation
{

    private readonly Variable _var1;
    private readonly Variable _var2;

    public Multiply(Variable var1, Variable var2)
    {
        _var1 = var1;
        _var2 = var2;
    }

    public double GetValue()
    {
        return _var1.Value() * _var2.Value();
    }

    public double GetValues(Dictionary<Variable, double> valueStore)
    {
        if(!valueStore.ContainsKey(_var1))
            _var1.Values(valueStore);
        
        if(!valueStore.ContainsKey(_var2))
            _var2.Values(valueStore);

        return valueStore[_var1] * valueStore[_var2];
    }

    public Variable[] GetDependencies()
    {
        return new Variable[] { _var1, _var2 };
    }

    public double Derive(Variable var, Dictionary<Variable, double> values)
    {
        if (var == _var1)
            return values[_var2];
        else if (var == _var2)
            return values[_var1];

        return 0;
    }
}


public class Add : IOperation
{


    private readonly Variable _var1;
    private readonly Variable _var2;

    public Add(Variable var1, Variable var2)
    {
        _var1 = var1;
        _var2 = var2;
    }
    
    public double GetValue()
    {
        return _var1.Value() + _var2.Value();
    }

    public double GetValues(Dictionary<Variable, double> valueStore)
    {
        if(!valueStore.ContainsKey(_var1))
            _var1.Values(valueStore);
        
        if(!valueStore.ContainsKey(_var2))
            _var2.Values(valueStore);

        return valueStore[_var1] + valueStore[_var2];
    }

    public Variable[] GetDependencies()
    {
        return new Variable[] { _var1, _var2 };
    }

    public double Derive(Variable var, Dictionary<Variable, double> values)
    {
        return (var == _var1 || var == _var2) ? 1 : 0;
    }
    
}


public class Invert : IOperation
{

    
    private readonly Variable _var;

    public Invert(Variable var)
    {
        _var = var;
    }
    
    public double GetValue()
    {
        return 1 / _var.Value();
    }

    public double GetValues(Dictionary<Variable, double> valueStore)
    {
        if(!valueStore.ContainsKey(_var))
            _var.Values(valueStore);

        return 1 / valueStore[_var];
    }

    public Variable[] GetDependencies()
    {
        return new Variable[] { _var };
    }

    public double Derive(Variable var, Dictionary<Variable, double> values)
    {
        return -1 * System.Math.Pow(values[_var], -2);
    }
    
}