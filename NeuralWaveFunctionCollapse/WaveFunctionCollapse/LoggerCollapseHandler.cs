using NeuralWaveFunctionCollapse.Math;

namespace NeuralWaveFunctionCollapse.WaveFunctionCollapse;

public class LoggerCollapseHandler: ICollapseHandler
{


    // width x height x input
    private readonly Tensor<double> _output;

    private readonly Func<int, int, Tensor<int>, Tensor<double>, Tensor<double>> _getDataAt;

    private readonly Tensor<double> _decisionData;
    private readonly Tensor<int> _decisionLabels;

    private int _currentIndex = 0;

    public LoggerCollapseHandler(Tensor<double> output, Func<int, int, Tensor<int>, Tensor<double>, Tensor<double>> getDataAt, int kernelSize)
    {
        _output = output;
        _getDataAt = getDataAt;

        var width = output.GetShape().GetSizeAt(0);
        var height = output.GetShape().GetSizeAt(1);
        var inputCount = output.GetShape().GetSizeAt(2);

        _decisionData = new Tensor<double>(Shape.Of(width * height, kernelSize, inputCount));
        _decisionLabels = new Tensor<int>(Shape.Of(width * height));
    }
    
    
    public int Collapse(Tensor<double> probabilities, Tensor<int> collapsed, Tensor<double> input, int x, int y)
    {
        var result = (int) _output.GetValue(x, y, 0);

        _decisionData.Copy(_getDataAt(x, y, collapsed, input), _currentIndex);
        _decisionLabels.SetValue(result, _currentIndex);

        _currentIndex++;
        
        return result;
    }


    public Tensor<double> DecisionData => _decisionData;

    public Tensor<int> DecisionLabels => _decisionLabels;


    public void ResetHead()
    {
        _currentIndex = 0;
    }

}