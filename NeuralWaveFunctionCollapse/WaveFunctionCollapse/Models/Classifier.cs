using NeuralWaveFunctionCollapse.IO;
using NeuralWaveFunctionCollapse.Math;
using NeuralWaveFunctionCollapse.Math.AutoDif;
using NeuralWaveFunctionCollapse.Math.Optimisation;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;



public interface IWaveFunctionClassifier<TClassifierConfig>
{

    Tensor<Variable> Classify(Tensor<double> input);

    void TrainClassifier(Tensor<double> input, Tensor<int> labels, TClassifierConfig config);

    void Build(int kernelSize, int outputDimensions, int inputDimensions);

    void Build(int kernelSize, int outputDimensions, int inputDimensions, string file, IoManager ioManager);
    
    void Save(string file, IoManager ioManager);

}