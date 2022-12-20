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


        var mapCount = 1;
        var input = new Tensor<double>[mapCount];
        for (var i = 0; i < mapCount; i++)
        {
            var level = ioManager.Load<LdtkLevel>("D:/projects/NeuralWaveFunctionCollpase/NeuralWaveFunctionCollapse/Data/Level_" + 102 + ".ldtkl");
            input[i] = level.GetLayer("StructureLayer").ToTensor().UpDimension();    
        }
        
        var possibleOutputStates = 5;

        var network = 
            Network.Sequential(
                new DenseLayer(Shape.Of(30), Activation.ReLu),
                new DenseLayer(Shape.Of(30), Activation.ReLu),
                new DenseLayer(Shape.Of(possibleOutputStates), Activation.Identity)
            );
        
        var model = new ClassifierModel<NeuralNetworkTrainingConfig>(
            new NeuralNetworkClassifier(network), 
            2,
            possibleOutputStates);

        var config = new NeuralNetworkTrainingConfig() 
        {
            Epochs = 40,
            Optimiser = new StochasticGradientDescentOptimiser(new SgdConfig() {
                Iterations = 1,
                LearnRate = 0.00005
            }),
            Loss = MeanSquaredError.Of,
            TestRatio = 0.1
        };

        model.Build(input, 1, config, 5);

        
        var grid = new Grid(20, 20, possibleOutputStates, model, new RandomCollapseHandler(54684)); 
        grid.Collapse();  
        grid.GetOutput().Print(true);
    }
    
}