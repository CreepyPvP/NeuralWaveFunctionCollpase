namespace NeuralWaveFunctionCollapse.IO;



public interface IImporter
{
    
}

public interface IImporter<T>: IImporter
{

    T Load(string file);

}


public interface IExporter
{
    
}


public interface IExporter<T> : IExporter
{

    void Export(string file, T data);

}

public class IoManager
{

    private readonly Dictionary<Type, IImporter> _importers = new();

    private readonly Dictionary<Type, IExporter> _exporters = new();


    public void RegisterImporter<T>(IImporter<T> importer)
    {
        _importers[typeof(T)] = importer;
    }

    public T Load<T>(string file)
    {
        var importer = (IImporter<T>) _importers[typeof(T)];

        if (importer == null) throw new Exception("No importer registered for type");
        
        return importer.Load(file);
    }


    public void RegisterExporter<T>(IExporter<T> exporter)
    {
        _exporters[typeof(T)] = exporter;
    }

    public void Export<T>(string file, T data)
    {
        var exporter = (IExporter<T>) _exporters[typeof(T)];

        if (exporter == null) throw new Exception("No exporter registered for type");

        exporter.Export(file, data);
    } 
    
}