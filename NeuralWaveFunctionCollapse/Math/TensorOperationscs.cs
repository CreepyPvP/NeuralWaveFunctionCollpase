using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.Math;

public static class TensorOperations
{

    public static Tensor<double> Mul(this Tensor<double> self, Tensor<double> m, bool disableChecks = false)
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

        var output = new Tensor<double>(Shape.Sub(self.GetShape(), m.GetShape().GetDimensionality()));

        output.GetShape().ForEach(o =>
        {
            double v = 0;
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
    
    
}