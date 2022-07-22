using System.Diagnostics;

namespace NeuralWaveFunctionCollapse;


class BenchmarkStructure
{
    private static readonly int Width = 100;
    private static readonly int Height = 100;
    
    public static void Start()
    {

        var benchmark = new Benchmark.Benchmark(() =>
        {
            for (var i = 0; i < Width * Height; i++)
            {
                var lowest = double.MaxValue;
                var lowestX = -1;
                var lowestY = -1;
            
                for (var x = 0; x < Width; x++)
                {
                    for (var y = 0; y < Height; y++)
                    {
                    
                        var entropy = SimulateEntropyCalculation(x, y);
                        if (entropy < lowest)
                        {
                            lowestX = x;
                            lowestY = y;
                            lowest = entropy;
                        }
                    
                    }
                }    
            }
        }, 20);
        benchmark.Run();
        Console.WriteLine("Avg time: " + benchmark.AvgTime);
        Console.WriteLine("Total time: " + benchmark.TotalTime);
    }


    static double SimulateEntropyCalculation(int x, int y)
    {
        var xValue = x - (Width / 2);
        var yValue = y - (Height / 2);
        return xValue * xValue + yValue * yValue;
    }
    
}