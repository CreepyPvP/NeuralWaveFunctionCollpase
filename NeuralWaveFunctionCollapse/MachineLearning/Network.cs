using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.MachineLearning;

public class Network
{

    private readonly GraphNode<Layer> _graph;

    private Network(GraphNode<Layer> graph)
    {
        _graph = graph;
    }
    
    public static Network Sequential(Layer[] layers)
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