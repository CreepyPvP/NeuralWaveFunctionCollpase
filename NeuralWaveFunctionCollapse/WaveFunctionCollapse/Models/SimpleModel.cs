using System.Xml.Linq;
using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse.Models;

public class SimpleModel: IWaveFunctionModel
{

    
    public SimpleModel()
    {
        
    }
    
    public bool Impacts(int collapseX, int collapseY, int posX, int posY)
    {
        return System.Math.Abs(collapseX - posX) <= 1 && System.Math.Abs(collapseY - posY) <= 1;
    }

    // 0 cant be next to 2, 1 can be next to 0 and 2
    // void counts as 0
    public Tensor CalculateDistribution(int x, int y, DataContainer<int> collapsed, Tensor additionalData)
    {
        var canBe0 = true;
        var canBe2 = true;

        var gridWidth = collapsed.GetShape().GetSizeAt(0);
        var gridHeight = collapsed.GetShape().GetSizeAt(1);

        var output = new Tensor(Shape.Of(3));
        
        for (var dX = -1; dX <= 1; dX++)
        {
            for (var dY = -1; dY <= 1; dY++)
            {
                if(dX == 0 && dY == 0) continue;

                var posX = x + dX;
                var posY = y + dY;

                var value = (posX < 0 || posY < 0 || posX >= gridWidth || posY >= gridHeight)
                    ? 0
                    : collapsed.GetValue(posX, posY);

                canBe0 = canBe0 && value != 2;
                canBe2 = canBe2 && value != 0;
            }
        }

        output.SetValue(canBe0 ? 1 : 0, 0);
        output.SetValue(2, 1);
        output.SetValue(canBe2 ? 4 : 0, 2);
        
        // output.Print();
        
        return output;
    }
    
}