using NeuralWaveFunctionCollapse.Math.AutoDif;

namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

public static class Activation
{

    public static Variable Identity(Variable input)
    {
        return input;
    }


    public static Variable ReLu(Variable input)
    {
        return Variable.Max(input, 0);
    }
    
}