namespace NeuralWaveFunctionCollapse.MachineLearning.NeuralNetwork;


[Serializable]
public class SerializedNetwork
{

    public Dictionary<string, Layer> layers = new();

    public int period = 0;
    

    public SerializedNetwork()
    {
        
    }

}