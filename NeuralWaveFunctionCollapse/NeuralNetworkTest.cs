using System.Diagnostics;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Loss;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse;


class NeuralNetworkTest
{
    public static void Start()
    {

        var network = Network.Sequential(new DenseLayer(Shape.Of(1), Activation.Identity, "dense_layer"));

        network.Compile(Shape.Of(1));

        var optimiser = new StochasticGradientDescentOptimiser(new SgdConfig()
        {
            Iterations = 10,
            LearnRate = 0.01
        });
        
        var config = new NeuralNetworkTrainingConfig()
        {
            Epochs = 10,
            Loss = MeanSquaredError.Of,
            Optimiser = optimiser
        };

        var dataPointCount = 10;

        var data = new Tensor<double>(Shape.Of(dataPointCount, 1));
        var labels = new Tensor<double>(Shape.Of(dataPointCount));

        for (var i = 0; i < dataPointCount; i++)
        {
            data.SetValue(i, i, 0);
            labels.SetValue(i * 2, i);
        }
        
        network.Train(data, labels, config);



        var test = new Tensor<double>(Shape.Of(1));
        test.SetValue(3.5, 0);
        
        network.Simulate(test).Evaluate().Print();
    }

}