using NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;

namespace NeuralWaveFunctionCollapse.IO.Impl;

public class NetworkExporter: IExporter<Network>
{
    
    public async void Export(string file, Network data)
    {
        await File.WriteAllTextAsync(file, data.ToString());
    }
    
}