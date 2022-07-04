﻿using System.Runtime.InteropServices;
using NeuralWaveFunctionCollapse.Math;

Matrix input = new Matrix(Shape.Of(3));

Matrix weights = new Matrix(Shape.Of(3, 1));

input.SetValue(1, 0);
input.SetValue(2, 1);
input.SetValue(3, 2);

weights.SetValue(0.5, 0, 0);
weights.SetValue(2, 1, 0);
weights.SetValue(4, 2, 0);

Console.WriteLine(weights.Mul(input).GetValue(0));