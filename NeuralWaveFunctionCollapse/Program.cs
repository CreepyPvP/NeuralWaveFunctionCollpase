using NeuralWaveFunctionCollapse.Benchmark;
using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.IO.Impl;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Loss;
using NeuralWaveFunctionCollapse.MachineLearning.RandomForest;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;
using NeuralWaveFunctionCollapse.Types;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models.Classifiers;

namespace NeuralWaveFunctionCollapse;


/*
 * TODO:
 *
 * + Optimisation
 *      - Hash list for entropy searching
 *      - Object pooling
 *
 * + Features
 *      - Activation functions (Relu) (done)
 *      - Better wave function classier model data preparation
 *      - Saving / Loading models
 *      - Delayed model training
 *      - Add biases (done)
 *      - Add wave function collapse failure detection
 */
class Program
{

    public static void Main()
    {
        var ioManager = new IoManager();
        ioManager.RegisterImporter(new LdtkLevelImporter());
        
        var level = ioManager.Load<LdtkLevel>("C:/Users/inter/Desktop/maps/maps/Level_1.ldtkl");
        var input = level.GetLayer("StructureLayer").ToTensor().UpDimension();
        
        var possibleOutputStates = 5;

        var network = 
            Network.Sequential(
                new DenseLayer(Shape.Of(10), Activation.ReLu),
                new DenseLayer(Shape.Of(10), Activation.ReLu),
                new DenseLayer(Shape.Of(10), Activation.ReLu),
                new DenseLayer(Shape.Of(possibleOutputStates), Activation.Identity)
            );
        
        var model = new ClassifierModel<NeuralNetworkTrainingConfig>(
            new NeuralNetworkClassifier(network), 
            3,
            possibleOutputStates);

        var config = new NeuralNetworkTrainingConfig()
        {
            Epochs = 1000,
            Optimiser = new StochasticGradientDescentOptimiser(new SgdConfig() {
                Iterations = 1,
                LearnRate = 0.0004
            }),
            Loss = MeanSquaredError.Of
        };

        model.Build(input, config);



        var grid = new Grid(16, 12, possibleOutputStates, model, 64567); 
        grid.Collapse();  
        grid.GetOutput().Print(true);

        /*
        var test = new Tensor<double>(Shape.Of(24, 1));
        test.SetValue(0, 0, 0);
        test.SetValue(0, 1, 0);
        test.SetValue(1, 2, 0);
        test.SetValue(0, 3, 0);
        test.SetValue(0, 4, 0);

        test.SetValue(0, 5, 0);
        test.SetValue(0, 6, 0);
        test.SetValue(1, 7, 0);
        test.SetValue(0, 8, 0);
        test.SetValue(0, 9, 0);
        
        test.SetValue(0, 10, 0);
        test.SetValue(0, 11, 0);
        test.SetValue(0, 12, 0);
        test.SetValue(0, 13, 0);
        
        test.SetValue(0, 14, 0);
        test.SetValue(0, 15, 0);
        test.SetValue(2, 16, 0);
        test.SetValue(3, 17, 0);
        test.SetValue(3, 18, 0);
        
        test.SetValue(0, 19, 0);
        test.SetValue(0, 20, 0);
        test.SetValue(2, 21, 0);
        test.SetValue(0, 22, 0);
        test.SetValue(0, 23, 0);
        
        network.Simulate(test).Evaluate().Print();
        
        // Console.WriteLine("Avg time: " + benchmark.AvgTime);    // old system: 1,7709 (50x50 grid, 3 radius, 50 trees)
        */
    }
    
}