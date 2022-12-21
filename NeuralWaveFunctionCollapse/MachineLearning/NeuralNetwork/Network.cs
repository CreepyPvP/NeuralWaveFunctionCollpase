using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork.Benchmark;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;
using NeuralWaveFunctionCollapse.Util;
using NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;
using Newtonsoft.Json.Linq;

namespace NeuralWaveFunctionCollapse.MachineLearning. NeuralNetwork;



public struct NeuralNetworkTrainingConfig
{
    
    // Loss(output, labels)
    public Func<Tensor<Variable>, Tensor<double>, Variable> Loss;

    public IOptimiser Optimiser;

    public int Epochs;

    public double TestRatio;

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


    public void Compile(Shape input, string file, IoManager ioManager)
    {
        var data = ioManager.Load<JObject>(file);
        JToken[] layers = data["layers"].Children().ToArray();
        
        _inputShape = input;

        var random = new SeededRandom(4534509);
        
        _input = new InputDataSource(input);
        _graph.GetValue().Build(_input, random, FindLayer(layers, _graph.GetValue().Id));
        
        _graph.ForEach(layer =>
        {
            layer.GetChildren().ForEach(child => child.GetValue().Build(layer.GetValue(), random, FindLayer(layers, child.GetValue().Id)));

            if (layer.GetChildren().Count == 0) _output = layer.GetValue().GetValue();
        });
    }

    private JToken FindLayer(JToken[] layers, string id)
    {
        return layers.FirstOrDefault(token => (string)token["id"] == id);
    }
    
    public void Compile(Shape input)
    {
        _inputShape = input;

        var random = new SeededRandom(4534509);
        
        _input = new InputDataSource(input);
        _graph.GetValue().Build(_input, random, null);
        
        _graph.ForEach(layer =>
        {
            layer.GetChildren().ForEach(child => child.GetValue().Build(layer.GetValue(), random, null));

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

        var testSize = (int) System.Math.Min(config.TestRatio * data.Length, data.Length);
        var tests = new Tensor<double>[testSize];
        var testLabels = new Tensor<double>[testSize];

        var trainingBenchmark = new SimpleTrainingBenchmark();

        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            random.Shuffle(indices);

            // training
            for (var i = 0; i < data.Length - testSize; i++)
            {
                var index = indices[i];

                config.Optimiser.Minimize(config.Loss(Simulate(data[index], true), labelArr[index]), trainingBenchmark);   
            }
            
            // testing
            for (var i = 0; i < testSize; i++)
            {
                tests[i] = data[data.Length - testSize + i];
                testLabels[i] = labelArr[data.Length - testSize + i];
            }
            if(testSize > 0)
                Evaluate(tests, testLabels, config.Loss);
        }
    }

    
    // input: index x datapoint
    // labels: index x outputs
    public void Evaluate(Tensor<double> data, Tensor<double> labels, Func<Tensor<Variable>, Tensor<double>, Variable> loss)
    {
        Evaluate(data.ToArray(), labels.ToArray(), loss);
    }

    public void Evaluate(Tensor<double>[] data, Tensor<double>[] labels, Func<Tensor<Variable>, Tensor<double>, Variable> loss)
    {
        var trainingBenchmark = new SimpleTrainingBenchmark();

        for (var i = 0; i < data.Length; i++)
        {
            trainingBenchmark.PushResult(loss(Simulate(data[i], true), labels[i]).Value());
        }

        trainingBenchmark.EndEpoch();
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


    public override string ToString()
    {
        var layers = new List<Layer>();
        _graph.ForEach(layer => layers.Add(layer.GetValue()));
        var serializedLayers = string.Join(", ", layers.Select(layer => layer.ToString()));
        return "{\"layers\": [" + serializedLayers + "]}";
    }
}