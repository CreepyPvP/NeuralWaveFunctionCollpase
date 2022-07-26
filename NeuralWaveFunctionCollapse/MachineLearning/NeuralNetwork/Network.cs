using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse.MachineLearning. NeuralNetwork;



public readonly struct NeuronalNetworkTrainingConfiguration<TOptimiserConfiguration>
{

    public readonly Func<Tensor<double>, Tensor<double>, Variable> Loss;

    public readonly IOptimiser<TOptimiserConfiguration> Optimiser;

}


public class Network: IClassifier<>
{

    private readonly GraphNode<Layer> _graph;

    private InputDataSource _input;
    private IDataSource _output;
    
    
    private Network(GraphNode<Layer> graph)
    {
        _graph = graph;
    }

    public void Compile(Shape input)
    {
        _input = new InputDataSource(input);
        _graph.GetValue().RegisterInput(_input);
        
        _graph.ForEach(layer =>
        {
            layer.GetChildren().ForEach(child => child.GetValue().RegisterInput(layer.GetValue()));
            
            // TODO: add proper output neuron selection (maybe by id??)
            if (layer.GetChildren().Count == 0) _output = layer.GetValue();
        });
    }

    public Tensor<double> Simulate(Tensor<double> input, bool training = true, bool disableChecks = false)
    {
        _output.Flush();

        _input.SetInput(input, disableChecks);

        return _output.GetData();
    }
    
    public static Network Sequential(params Layer[] layers)
    {
        GraphNode<Layer>? root = null;

        if (layers.Length == 0) throw new Exception("Cannot create a network from 0 layers");
        
        foreach (var layer in layers)
        {
            var current = new GraphNode<Layer>(layer);
            if (root == null) root = current;
        }

        return new Network(root!);
    }

    public Tensor<double> Classify(Tensor<double> input)
    {
        throw new NotImplementedException();
    }

    public void Train(Tensor<double> input, Tensor<int> labels, NeuronalNetworkTrainingConfiguration configuration)
    {
        throw new NotImplementedException();
    }


}