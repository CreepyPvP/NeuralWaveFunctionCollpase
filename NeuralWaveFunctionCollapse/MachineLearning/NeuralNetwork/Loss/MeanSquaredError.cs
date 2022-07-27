using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Loss;

public static class MeanSquaredError
{

    public static Variable Of(Tensor<Variable> output, Tensor<double> label)
    {
        var diff = output.Compare(label, (cOutput, cLabel) => cOutput - cLabel);

        var result = Variable.Of(0, false);
        
        diff.ForEach(var =>
        {
            result += var * var;
        });
        
        // Console.WriteLine(result.Value());

        return result;
    }
    
}