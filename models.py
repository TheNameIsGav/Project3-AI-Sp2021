import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions) #given

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w #for these the Parameter model IS the weights of the model

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        return nn.DotProduct(x, self.get_weights()) #gets the dot product of the feature vector (x), and the weights of x given by get_weights

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        prediction = nn.as_scalar(self.run(x)) #converts the given node to a dot product, then to scalar
        if prediction >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        for _ in range(501): #501 garuntees passing accuracy
            for x, y in dataset.iterate_once(1): #going to two increases parameters
                prediction = self.get_prediction(x) #gets -1 or 1
                if prediction != nn.as_scalar(y): #if what we think is not what it is
                    self.get_weights().update(x, nn.as_scalar(y)) #update the weights of that x

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 20
        self.featuresLayer1 = 200
        self.featuresLayer2 = 1
        self.featuresLayer3 = 1

        #generate a bunch of parameter nodes to help with finer generation
        self.weight1 = nn.Parameter(1, self.featuresLayer1)
        self.weight2 = nn.Parameter(self.featuresLayer1, self.featuresLayer2)
        self.weight3 = nn.Parameter(self.featuresLayer2,self.featuresLayer3)
        self.batch1= nn.Parameter(1,self.featuresLayer1)
        self.batch2 = nn.Parameter(1,self.featuresLayer2)
        self.batch3 = nn.Parameter(1,self.featuresLayer3)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        #generate multiple, more refined layers as we work through the model
        feature1 = nn.Linear(x, self.weight1)
        layer1 = nn.AddBias(feature1,self.batch1)
        layer1 = nn.ReLU(layer1)
        feature2 = nn.Linear(layer1, self.weight2)
        layer2 = nn.AddBias(feature2, self.batch2)
        return layer2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictY = self.run(x)
        loss = nn.SquareLoss(predictY, y) #dummy thick square loss function given to us
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epsilon = -0.006 #learning rate
        totalLoss = 0 
        totalSamples = 0

        for x,y in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(x,y)
            weight1Gradient, batch1Gradient, weight2Gradient, batch2Gradient= nn.gradients(loss,[self.weight1, self.batch1, self.weight2, self.batch2])

            #update the weights and the batchs
            self.weight1.update(weight1Gradient, epsilon)
            self.batch1.update(batch1Gradient, epsilon)

            self.weight2.update(weight2Gradient, epsilon)
            self.batch2.update(batch2Gradient, epsilon)

            #update the total loss, samples, and find the average loss
            totalLoss += nn.as_scalar(loss)*self.batchSize
            totalSamples += self.batchSize
            average_loss = totalLoss/totalSamples

            #when we have completed the training
            if(average_loss)<0.02:
                break

        return

#we've done this before in functional
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 40
        self.featureLayer1 = 300
        self.featureLayer2 = 400
        self.featureLayer3 = 10

        self.weight1 = nn.Parameter(784, self.featureLayer1)
        self.weight2 = nn.Parameter(self.featureLayer1, self.featureLayer2)
        self.weight3 = nn.Parameter(self.featureLayer2, self.featureLayer3)
        self.batch1 = nn.Parameter(1, self.featureLayer1)
        self.batch2 = nn.Parameter(1, self.featureLayer2)
        self.batch3 = nn.Parameter(1, self.featureLayer3)

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.
        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        feature1 = nn.Linear(x, self.weight1)
        layer1 = nn.AddBias(feature1, self.batch1)
        layer1 = nn.ReLU(layer1)
        feature2 = nn.Linear(layer1, self.weight2)
        layer2 = nn.AddBias(feature2, self.batch2)
        layer2 = nn.ReLU(layer2)
        feature3 = nn.Linear(layer2, self.weight3)
        layer3 = nn.AddBias(feature3, self.batch3)
        return layer3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictY = self.run(x)
        loss = nn.SoftmaxLoss(predictY, y) #this function does something useful, idk it had loss in it and couldn't use square loss like before
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.05
        iteration = 1
        for x, y in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(x, y)
            weight1Grad, batch1Grad, weight2Grad, batch2Grad, weight3Grad, batch3Grad = nn.gradients(loss,[self.weight1, self.batch1, self.weight2, self.batch2, self.weight3, self.batch3])

            self.weight1.update(weight1Grad, learning_rate)
            self.batch1.update(batch1Grad, learning_rate)

            self.weight2.update(weight2Grad, learning_rate)
            self.batch2.update(batch2Grad, learning_rate)

            self.weight3.update(weight3Grad, learning_rate)
            self.batch3.update(batch3Grad, learning_rate)

            if iteration%25 == 0:
                accuracy = dataset.get_validation_accuracy()
                if accuracy>0.98:
                    break
            iteration +=1
        return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batchSize = 40
        self.literal0 = 47
        self.literal1 = 256
        self.literal2 = 256
        self.literal3 = 500
        self.literal4 = 5


        self.weight1 = nn.Parameter(self.literal0,self.literal1)
        self.batch1 = nn.Parameter(1,self.literal1)

        self.weight2 = nn.Parameter(self.literal1,self.literal2)
        self.batch2 = nn.Parameter(1,self.literal2)

        self.weight3 = nn.Parameter(self.literal2, self.literal3)
        self.batch3 = nn.Parameter(1,self.literal3)

        self.weight4 = nn.Parameter(self.literal3, self.literal4)
        self.batch4 = nn.Parameter(1,self.literal4)

    def run(self, xs):
        """
        Runs the model for a batch of examples.
        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        featureInit = nn.Linear(xs[0], self.weight1)
        previousOutput = featureInit
        for i in range(1, len(xs)):
            feature = nn.Add(nn.Linear(xs[i],self.weight1), nn.Linear(previousOutput,self.weight2))
            previousOutput = feature
        feature2 = nn.ReLU(nn.AddBias(nn.Linear(previousOutput,self.weight3), self.batch3))
        ferature3 = nn.AddBias(nn.Linear(feature2,self.weight4), self.batch4)

        return ferature3

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictY = self.run(xs)
        loss = nn.SoftmaxLoss(predictY, y) #same as before
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epsilon = -0.05
        iteration = 1
        for x, y in dataset.iterate_forever(self.batchSize):
            loss = self.get_loss(x, y)

            w1Grad, w2Grad, w3Grad, b3Grad, w4Grad, b4Grad = nn.gradients(loss,[self.weight1, self.weight2, self.weight3, self.batch3, self.weight4, self.batch4])
            
            self.weight1.update(w1Grad, epsilon)

            self.weight2.update(w2Grad, epsilon)

            self.weight3.update(w3Grad, epsilon)
            self.batch3.update(b3Grad, epsilon)

            self.weight4.update(w4Grad, epsilon)
            self.batch4.update(b4Grad, epsilon)

            if iteration % 25 == 0:
                accuracy = dataset.get_validation_accuracy()
                if accuracy > 0.85:
                    break
            iteration += 1
        return