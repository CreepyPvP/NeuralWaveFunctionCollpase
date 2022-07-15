using NeuralWaveFunctionCollapse.MachineLearning.RandomForest;
using NeuralWaveFunctionCollapse.Math;

var forest = new RandomForestClassifier();


var trainingSize = 40;

var trainingData = new Tensor(Shape.Of(trainingSize * 2, 2));
var labels = new DataContainer<int>(Shape.Of(trainingSize * 2));

var i = 0;

// class 0
for (; i < trainingSize; i++)
{
    trainingData.SetValue(i, i, 0);
    trainingData.SetValue(i * i,i, 1);
    labels.SetValue(0, i);
}

// class 1
for (; i < trainingSize * 2; i++)
{
    trainingData.SetValue(i - trainingSize, i, 0);
    trainingData.SetValue(0.5 * (i - trainingSize),i, 1);
    labels.SetValue(1, i);
}

forest.Train(trainingData, labels, 40, 453489);



var test = new Tensor(Shape.Of(2));
test.SetValue(12.4, 0);
test.SetValue(12.4*0.5, 1);

Console.WriteLine("Prediction: " + forest.Predict(test));