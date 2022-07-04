using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public class Network
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

    public Tensor Simulate(Tensor input, bool training = true, bool disableChecks = false)
    {
        _output.Flush();

        _input.SetInput(input);

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
    
}