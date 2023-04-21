import enum
import nn
import pdb

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dot = nn.as_scalar(self.run(x))
        return 1 if dot >= 0 else -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        #pdb.set_trace()
        while True:
            done = True
            for x, y in dataset.iterate_once(1):
                const_y = nn.as_scalar(y)
                if self.get_prediction(x) != const_y:
                    self.w.update(x, const_y)
                    done = False
                    break
            if done:
                break
                


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        HL_SIZE = 400
        self.LR = .02

        self.F0 = nn.Parameter(1, HL_SIZE)
        self.b0 = nn.Parameter(1, HL_SIZE)
        self.M0 = nn.Parameter(HL_SIZE, HL_SIZE)

        self.F1 = nn.Parameter(1, HL_SIZE)
        self.b1 = nn.Parameter(1, HL_SIZE)
        self.M1 = nn.Parameter(HL_SIZE, HL_SIZE)
        
        self.M2 = nn.Parameter(HL_SIZE, HL_SIZE)
        self.M3 = nn.Parameter(HL_SIZE, HL_SIZE)
        self.M4 = nn.Parameter(HL_SIZE, HL_SIZE)
        self.M5 = nn.Parameter(HL_SIZE, HL_SIZE)
        
        self.b2 = nn.Parameter(1, HL_SIZE)
        self.b3 = nn.Parameter(1, HL_SIZE)

        self.R = nn.Parameter(HL_SIZE, 1)
        self.params = [self.F0, self.F1, self.b0, self.b1, self.b2, self.b3, self.M0, self.M1, self.M2, self.M3, self.M4, self.M5]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        features = nn.Linear(x, self.F0)
        features = nn.Linear(features, self.M0)
        features = nn.AddBias(features, self.b0)
        features = nn.ReLU(features)
        features = nn.Linear(features, self.M4)

        features2 = nn.Linear(x, self.F1)
        features2 = nn.Linear(features2, self.M1)
        features2 = nn.AddBias(features2, self.b1)
        features2 = nn.ReLU(features2)
        features2 = nn.Linear(features2, self.M5)

        layer = nn.Add(features, features2)
        layer = nn.AddBias(layer, self.b2)
        layer = nn.Linear(layer, self.M2)
        layer = nn.ReLU(layer)
        layer = nn.Linear(layer, self.M3)
        layer = nn.AddBias(layer, self.b3)
        return nn.Linear(layer, self.R)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        data_size = dataset.x.shape[0]
        batch_size = 2
        for i in range(9, int(pow(data_size, .5)) + 1):
            if data_size % i == 0:
                batch_size = i
        
        xF = 0
        yF = 0
        for x, y in dataset.iterate_once(data_size):
            xF = x
            yF = y

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            lossF = self.get_loss(xF, yF)
            if nn.as_scalar(lossF) < .02:
                print(nn.as_scalar(loss))
                break
            #pdb.set_trace()
            gradients = nn.gradients(loss, self.params)
            for i, param in enumerate(self.params):
                param.update(gradients[i], -1*self.LR)


        


        

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
        HL_SIZE = 128
        self.LR = .08

        self.M0 = nn.Parameter(784, HL_SIZE)
        self.b0 = nn.Parameter(1, HL_SIZE)

        self.M1 = nn.Parameter(HL_SIZE, 64)
        self.b1 = nn.Parameter(1, 64)
        
        self.M2 = nn.Parameter(64, 64)
        self.b2 = nn.Parameter(1, 64)

        self.M3 = nn.Parameter(64, 10)
        self.b3 = nn.Parameter(1, 10)

        self.params = [self.b0, self.b1, self.b2, self.b3, self.M0, self.M1, self.M2, self.M3]

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
        layer = nn.Linear(x, self.M0)
        layer = nn.AddBias(layer, self.b0)
        layer = nn.ReLU(layer)

        layer = nn.Linear(layer, self.M1)
        layer = nn.AddBias(layer, self.b1)
        layer = nn.ReLU(layer)

        layer = nn.Linear(layer, self.M2)
        layer = nn.AddBias(layer, self.b2)
        layer = nn.ReLU(layer)

        layer = nn.Linear(layer, self.M3)
        layer = nn.AddBias(layer, self.b3)

        return layer
        

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        data_size = dataset.x.shape[0]
        batch_size = 1
        for i in range(9, int(pow(data_size, .5)) + 1):
            if data_size % i == 0:
                batch_size = i

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            if dataset.get_validation_accuracy() > .975:
                break
            #pdb.set_trace()
            gradients = nn.gradients(loss, self.params)
            for i, param in enumerate(self.params):
                param.update(gradients[i], -1*self.LR)

        
        

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

        self.LR = .03
        # Initialize your model parameters here
        self.MH = nn.Parameter(5, 5)
        self.bH = nn.Parameter(1, 5)

        self.M0 = nn.Parameter(47, 128)
        self.b0 = nn.Parameter(1, 128)

        #self.M1 = nn.Parameter(47, 32)
        #self.b1 = nn.Parameter(1, 32)

        self.M1 = nn.Parameter(128, 5)
        self.b1 = nn.Parameter(1, 5)

        self.params = [self.b0, self.b1, self.M0, self.M1, self.MH, self.bH]

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
        word_len = len(xs)
        h_i = nn.Linear(xs[0], self.M0)
        h_i = nn.AddBias(h_i, self.b0)
        h_i = nn.ReLU(h_i)
        h_i = nn.Linear(h_i, self.M1)
        h_i = nn.AddBias(h_i, self.b1)
        for i in range(1, word_len):
            h = nn.Linear(xs[i], self.M0)
            h = nn.AddBias(h, self.b0)
            h = nn.ReLU(h)
            h = nn.Linear(h, self.M1)
            h = nn.AddBias(h, self.b1)
            h = nn.Add(h, nn.AddBias(nn.Linear(h_i, self.MH), self.bH))
            h_i = h
        return h_i
        



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
        return nn.SoftmaxLoss(self.run(xs), y)
        

    def train(self, dataset):
        """
        Trains the model.
        """

        #data_size = dataset.y.shape[0]
        #batch_size = 1
        #for i in range(9, int(pow(data_size, .5)) + 1):
        #    if data_size % i == 0:
        #        batch_size = i
        batch_size = 10
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            if dataset.get_validation_accuracy() > .81:
                break
            #pdb.set_trace()
            gradients = nn.gradients(loss, self.params)
            for i, param in enumerate(self.params):
                param.update(gradients[i], -1*self.LR)


        
