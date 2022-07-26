using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.MachineLearning. NeuralNetwork;



public readonly struct NeuronalNetworkTrainingConfig
{
    
    // Loss(output, labels)
    public readonly Func<Tensor<Variable>, Tensor<double>, Variable> Loss;

    public readonly IOptimiser Optimiser;

    public readonly int Epochs;

}


public class Network: IClassifier<NeuronalNetworkTrainingConfig>
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
        
        _input = new InputDataSource(input);
        _graph.GetValue().Build(_input);
        
        _graph.ForEach(layer =>
        {
            layer.GetChildren().ForEach(child => child.GetValue().Build(layer.GetValue()));

            if (layer.GetChildren().Count == 0) _output = layer.GetValue().GetValue();
        });
    }

    public Tensor<double> Simulate(Tensor<double> input, bool disableChecks = false)
    {
        _input.SetInput(input, disableChecks);

        return _output.Evaluate();
    }
    
    public static Network Sequential(params Layer[] layers)
    {
        GraphNode<Layer>? root = null;

        if (layers.Length == 0) throw new Exception("Cannot create a network from 0 layers");
        
        foreach (var layer in layers)
        {
            var current = new GraphNode<Layer>(layer);
            root ??= current;
        }

        return new Network(root!);
    }

    public Tensor<Variable> Classify(Tensor<double> input)
    {
        _input.SetInput(input);
        return _output;
    }

    
    // input: index x datapoint
    // label: index
    public void TrainClassifier(Tensor<double> input, Tensor<int> labels, NeuronalNetworkTrainingConfig config)
    {
        var data = input.ToArray();
        var labelArr = labels.ToArray();
        
        var random = new SeededRandom(43345);

        var indices = new int[data.Length];
        for (var i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        var expectedOutput = new Tensor<double>(_inputShape, 0);
        var prevClass = 0;
        
        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            random.Shuffle(indices);

            for (var i = 0; i < data.Length; i++)
            {
                var index = indices[i];

                var expectedClass = labelArr[index].GetValue(0);
                
                expectedOutput.SetValue(0, prevClass);
                expectedOutput.SetValue(1, expectedClass);
                prevClass = expectedClass;
                
                config.Optimiser.Minimize(config.Loss(Classify(data[index]), expectedOutput));   
            }
        }
    }

}