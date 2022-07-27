using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Util;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models.Classifiers;

public class NeuralNetworkClassifier: IWaveFunctionClassifier<NeuralNetworkTrainingConfig>
{

    private readonly Network _network;

    private int _outputDimensions;

        
    public NeuralNetworkClassifier(Network network)
    {
        _network = network;
    }
    
    public Tensor<Variable> Classify(Tensor<double> input)
    {
        return _network.Simulate(input, true);
    }


    public void Build(int kernelSize, int outputDimensions, int inputDimensions)
    {
        _outputDimensions = outputDimensions;
        _network.Compile(Shape.Of(kernelSize, inputDimensions));
    }
    
    // input: index x datapoint
    // label: index
    public void TrainClassifier(Tensor<double> input, Tensor<int> labels, NeuralNetworkTrainingConfig config)
    {
        var expectedNetworkOutput = new Tensor<double>(Shape.Of(labels.GetShape().GetSizeAt(0), _outputDimensions));

        for (var i = 0; i < labels.GetShape().GetSizeAt(0); i++)
        {
            var datapoint = new Tensor<double>(Shape.Of(_outputDimensions));
            for (var index = 0; index < _outputDimensions; index++)
            {
                datapoint.SetValue(labels.GetValue(i) == index? 1 : 0, index);
            }
            expectedNetworkOutput.Copy(datapoint, expectedNetworkOutput.GetShape().GetIndex(i, 0));
        }

        _network.Train(input, expectedNetworkOutput, config);
    }

}