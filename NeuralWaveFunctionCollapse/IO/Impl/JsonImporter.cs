using Newtonsoft.Json.Linq;

namespace NeuralWaveFunctionCollapse.IO.Impl;

public class JsonImporter: IImporter<JObject>
{
    public JObject Load(string file)
    {
        var data = File.ReadAllText(file);
        return JObject.Parse(data);
    }
    
}