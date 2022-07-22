using NeuralWaveFunctionCollapse.Benchmark;
using NeuralWaveFunctionCollapse.MachineLearning.RandomForest;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

namespace NeuralWaveFunctionCollapse;

class Program
{

    public static void Main()
    {
        var forest = new RandomForestClassifier(50, 45345);

        var model = new ClassifierModel(forest, 3);

        var possibleOutputStates = 10;

        var input = new Tensor(Shape.Of(20, 20, 1), 0);

        model.Build(input, possibleOutputStates);



        var benchmark = new Benchmark.Benchmark(() =>
        {
            var grid = new Grid(50, 50, possibleOutputStates, model, 55960589); 
            grid.Collapse();    
        }, 20);

        benchmark.Run();

        Console.WriteLine("Avg time: " + benchmark.AvgTime);    // old system: 1,7709 (50x50 grid, 3 radius, 50 trees)
    }
    
}