using NeuralWaveFunctionCollapse.Benchmark;
using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.IO.Impl;
using NeuralWaveFunctionCollapse.MachineLearning.RandomForest;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Types;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

namespace NeuralWaveFunctionCollapse;

class Program
{

    public static void Main()
    {
        var ioManager = new IoManager();
        ioManager.RegisterImporter(new LdtkLevelImporter());

        var level = ioManager.Load<LdtkLevel>("C:/Users/Luis/Desktop/maps/maps/Level_1.ldtkl");

        var input = level.GetLayer("StructureLayer").ToTensor().UpDimension();

        
        var forest = new RandomForestClassifier(100, 5456);

        var model = new ClassifierModel(forest, 2);

        var possibleOutputStates = 5;

        model.Build(input, possibleOutputStates);



        var grid = new Grid(16, 12, possibleOutputStates, model, 64567); 
        grid.Collapse();  
        grid.GetOutput().Print(true);
        
        // Console.WriteLine("Avg time: " + benchmark.AvgTime);    // old system: 1,7709 (50x50 grid, 3 radius, 50 trees)
        
    }
    
}