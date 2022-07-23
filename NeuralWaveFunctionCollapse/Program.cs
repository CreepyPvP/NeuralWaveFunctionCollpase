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



        // var grid = new Grid(16, 12, possibleOutputStates, model, 64567); 
        // grid.Collapse();  
        // grid.GetOutput().Print(true);


        var test = new Tensor(Shape.Of(24));
        test.SetValue(0, 0);
        test.SetValue(0, 1);
        test.SetValue(1, 2);
        test.SetValue(0, 3);
        test.SetValue(0, 4);

        test.SetValue(0, 5);
        test.SetValue(0, 6);
        test.SetValue(1, 7);
        test.SetValue(0, 8);
        test.SetValue(0, 9);
        
        test.SetValue(0, 10);
        test.SetValue(0, 11);
        test.SetValue(0, 12);
        test.SetValue(0, 13);
        
        test.SetValue(0, 14);
        test.SetValue(0, 15);
        test.SetValue(2, 16);
        test.SetValue(3, 17);
        test.SetValue(3, 18);
        
        test.SetValue(0, 19);
        test.SetValue(0, 20);
        test.SetValue(2, 21);
        test.SetValue(0, 22);
        test.SetValue(0, 23);
        
        forest.Classify(test).Print();
        
        // Console.WriteLine("Avg time: " + benchmark.AvgTime);    // old system: 1,7709 (50x50 grid, 3 radius, 50 trees)

    }
    
}