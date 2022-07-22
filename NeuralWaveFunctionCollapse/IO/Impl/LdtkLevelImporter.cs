using NeuralWaveFunctionCollapse.Types;
using Newtonsoft.Json;

namespace NeuralWaveFunctionCollapse.IO.Impl;

public class LdtkLevelImporter: IImporter<LdtkLevel>
{
    
    public LdtkLevel Load(string file)
    {
        var data = File.ReadAllText(file);
        if (data == null)
            throw new Exception("Couldn't read file " + file);
        
        var result = JsonConvert.DeserializeObject<LdtkLevel>(data);
        if (result == null)
            throw new Exception("Couldn't parse file " + file);

        return result;
    }
    
}