using System.Runtime.InteropServices;
using NeuralWaveFunctionCollapse.MachineLearning;
using NeuralWaveFunctionCollapse.MachineLearning.Layers;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

var network = Network.Sequential(new DenseLayer(Shape.Of(1)));

network.Compile(Shape.Of(3));

var input = new Tensor(Shape.Of(3));
input.SetValue(1, 0);
input.SetValue(2, 1);
input.SetValue(3, 2);

var result = network.Simulate(input);

Console.WriteLine(result.GetValue(0));  // should output 12.5

result = network.Simulate(input);
Console.WriteLine(result.GetValue(0));  // should output 12.5 again, to the the flush method



var outputElements = new int[] {0, 1, 2};
var seed = 10434;

var model = new SimpleModel();
var grid = new Grid<int>(30, 8, outputElements, model, seed);

Console.WriteLine("Grid Output: -----------------------------------------------");
grid.Collapse();
grid.GetOutput().Print();