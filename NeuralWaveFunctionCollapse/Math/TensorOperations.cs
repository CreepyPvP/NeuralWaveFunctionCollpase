using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.Math;

public static class TensorOperations
{

    public static Tensor<Variable> Mul(this Tensor<Variable> self, Tensor<Variable> m, bool disableChecks = false)
    {
        // equivalent to self x m
        
        if (!disableChecks)
        {
            var isValidOperation = !(self.GetShape().GetDimensionality() <= m.GetShape().GetDimensionality());

            for (var i = 0; i < m.GetShape().GetDimensionality() && isValidOperation; i++)
            {
                if (m.GetShape().GetLength(i) != self.GetShape().GetLength(i)) isValidOperation = false;
            }

            if (!isValidOperation) throw new Exception("Invalid Tensor multiplication");
        }

        var output = new Tensor<Variable>(Shape.Sub(self.GetShape(), m.GetShape().GetDimensionality()));

        output.GetShape().ForEach(o =>
        {
            var v = Variable.Of(0, false);
            m.GetShape().ForEach(k =>
            {
                var position = k.ArrJoin(o);
                v += m.GetValue(k) * self.GetValue(position);
            });
            output.SetValue(v, output.GetShape().GetIndex(o));
        });

        return output;
    }
    
    
    public static double GetLastLengthSquared(this Tensor<double> self, int size, params int[] start)
    {
        double sum = 0;
        var startIndex = self.GetShape().GetIndex(start);

        var values = self.GetRaw();
        
        for (var i = 0; i < size; i++)
        {
            sum += System.Math.Pow(values[startIndex + i], 2);
        }

        return sum;
    }


    public static Tensor<double> Evaluate(this Tensor<Variable> self)
    {
        var result = new Tensor<double>(self.GetShape());

        var values = new Dictionary<Variable, double>();
        
        self.GetShape().ForEach(pos =>
        {
            var variable = self.GetValue(pos);
            variable.Values(values);
            result.SetValue(values[variable], pos);
        });

        return result;
    }


}