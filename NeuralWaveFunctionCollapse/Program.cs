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
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
        ioManager.RegisterExporter(new NetworkExporter());
        ioManager.RegisterImporter(new JsonImporter());
        
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
                new DenseLayer(Shape.Of(15), Activation.ReLu, "dense_layer_0"),
                new DenseLayer(Shape.Of(15), Activation.ReLu, "dense_layer_1"),
                new DenseLayer(Shape.Of(possibleOutputStates), Activation.Identity, "dense_layer_2")
            );
        
        var model = new ClassifierModel<NeuralNetworkTrainingConfig>(
            new NeuralNetworkClassifier(network), 
            2,
            possibleOutputStates);

        var config = new NeuralNetworkTrainingConfig() 
        {
            Epochs = 10,
            Optimiser = new StochasticGradientDescentOptimiser(new SgdConfig() {
                Iterations = 1,
                LearnRate = 0.0005
            }),
            Loss = MeanSquaredError.Of,
            TestRatio = 0.1
        };

        model.Build(
            input,
            1, 
            config,
             300, 
            ioManager, 
            "D:/projects/NeuralWaveFunctionCollpase/NeuralWaveFunctionCollapse/Models/model",
            "D:/projects/NeuralWaveFunctionCollpase/NeuralWaveFunctionCollapse/Models/model-4.json");

        var grid = new Grid(20, 20, possibleOutputStates, model, new RandomCollapseHandler(54684)); 
        grid.Collapse();  
        grid.GetOutput().Print(true);
    }

    
}