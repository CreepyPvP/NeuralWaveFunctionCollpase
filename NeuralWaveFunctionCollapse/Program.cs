using System.Runtime.InteropServices;
using NeuralWaveFunctionCollapse.MachineLearning;
using NeuralWaveFunctionCollapse.MachineLearning.Layers;
using NeuralWaveFunctionCollapse.Math;

var network = Network.Sequential(new DenseLayer(Shape.Of(1)));

network.Compile(Shape.Of(3));

var input = new Tensor(Shape.Of(3));
input.SetValue(1, 0);
input.SetValue(2, 1);
input.SetValue(3, 2);

var result = network.Simulate(input);

Console.WriteLine(result.GetValue(0));  // should output 12.5