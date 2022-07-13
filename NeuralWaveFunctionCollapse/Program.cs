using System.Runtime.InteropServices;
using NeuralWaveFunctionCollapse.Benchmark;
using NeuralWaveFunctionCollapse.MachineLearning;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;



var outputElements = new int[] {0, 1, 2};
var seed = 10434;

var benchmark = new Benchmark(() =>
{
    var model = new SimpleModel();
    var grid = new Grid<int>(50, 50, outputElements, model, seed);
    
    grid.Collapse();
}, 30);

benchmark.Run();
Console.WriteLine("Avg time: " + benchmark.GetAvgTime());