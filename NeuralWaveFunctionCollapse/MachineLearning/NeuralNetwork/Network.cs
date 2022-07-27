using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;
using NeuralWaveFunctionCollapse.Util;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

namespace NeuralWaveFunctionCollapse.MachineLearning. NeuralNetwork;



public struct NeuralNetworkTrainingConfig
{
    
    // Loss(output, labels)
    public Func<Tensor<Variable>, Tensor<double>, Variable> Loss;

    public IOptimiser Optimiser;

    public int Epochs;

}


public class Network
{

    private readonly GraphNode<Layer> _graph;

    private InputDataSource _input;
    private Shape _inputShape;

    private Tensor<Variable> _output;


    private Network(GraphNode<Layer> graph)
    {
        _graph = graph;
    }

    public void Compile(Shape input)
    {
        _inputShape = input;

        var random = new SeededRandom(4534509);
        
        _input = new InputDataSource(input);
        _graph.GetValue().Build(_input, random);
        
        _graph.ForEach(layer =>
        {
            layer.GetChildren().ForEach(child => child.GetValue().Build(layer.GetValue(), random));

            if (layer.GetChildren().Count == 0) _output = layer.GetValue().GetValue();
        });
    }

    public Tensor<Variable> Simulate(Tensor<double> input, bool disableChecks = false)
    {
        _input.SetInput(input, disableChecks);

        return _output;
    }


    // input: index x datapoint
    // labels: index x outputs
    public void Train(Tensor<double> input, Tensor<double> labels, NeuralNetworkTrainingConfig config) {
        var data = input.ToArray();
        var labelArr = labels.ToArray();
        
        var random = new SeededRandom(43345);

        var indices = new int[data.Length];
        for (var i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            random.Shuffle(indices);

            for (var i = 0; i < data.Length; i++)
            {
                var index = indices[i];

                config.Optimiser.Minimize(config.Loss(Simulate(data[index], true), labelArr[index]));   
            }
        }
    }
    
    public static Network Sequential(params Layer[] layers)
    {
        GraphNode<Layer>? root = null;

        if (layers.Length == 0) throw new Exception("Cannot create a network from 0 layers");

        GraphNode<Layer>? ptr = null;

        foreach (var layer in layers)
        {
            var current = new GraphNode<Layer>(layer);
            if (root == null || ptr == null)
            {
                root = current;
            }
            else
            {
                ptr.Add(current);
            }
            
            ptr = current;
        }

        return new Network(root!);
    }

}