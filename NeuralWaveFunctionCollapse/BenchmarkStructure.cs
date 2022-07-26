﻿using System.Diagnostics;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Layers;
using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse;


class BenchmarkStructure
{
    private static readonly int Width = 100;
    private static readonly int Height = 100;

    public static void Start()
    {

        var network = Network.Sequential(new DenseLayer(Shape.Of(1)));

        network.Compile(Shape.Of(3));

        var input = new Tensor<double>(Shape.Of(3));
        input.SetValue(1, 0);
        input.SetValue(0.5, 1);
        input.SetValue(2, 2);

        network.Simulate(input).Print();

    }

}