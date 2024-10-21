# from multiprocessing import Value
import numpy as np
import pickle


class network:
    """
    class to hold a neural network
    """

    def __init__(self, n_in: int, n_out: int):
        """
        Initializes the Network class.

        Parameters:
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        """
        self.layers = []  # list to contain all layers
        self.n_in = n_in
        self.n_out = n_out
        self.mutateable_layers = []  # list to index mutateable layers

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.

        Parameters:
        layer: Layer object to be added to the network.
        """
        self.layers.append(layer)
        temp = self.check_integrity()
        if temp == False:
            print("Removing Added Layer: ", layer.__class__.__name__)
            del self.layers[-1]
        else:
            if layer.mutateable_weights or layer.mutateable_biases:
                self.mutateable_layers.append(len(self.layers) - 1)

        # need to check layer matches last layer

    def remove_layer(self, index: int):
        """
        Removes the layer at the specified index.

        Parameters:
        index (int): Index of the layer to be removed.
        """
        del self.layers[index]

    def check_integrity(self):
        """
        Checks the integrity of the layers in the network.
        """
        if self.layers == []:
            return True
        else:
            X = np.random.randn(1, self.n_in)
            try:
                self.forward(X)

            except Exception as e:
                print("Network Failed Integrity Test")
                print(e)
                return False
            return True

    def forward(self, X):
        """
        Propagates the input forward through the network.

        Parameters:
        X: 2D array of a batch of input vectors.

        Returns:
        numpy.ndarray: The output of the network after the forward pass.
        """
        if np.shape(X)[1] != self.n_in:
            raise Exception(f"Wrong input size: {np.shape(X)[1] }!={self.n_in}")
        else:
            self.output = np.array(X, dtype=float)  # make sure it in float format
            for layer in self.layers:
                self.output = layer.forward(self.output)
            return self.output

    def reset(self):  # set all layers to 0
        """
        Resets all mutable layers of the network.
        """
        for index in self.mutateable_layers:
            self.layers[index].reset()

    def random_initilisation(self, func=np.random.normal):
        """
        Initializes the network with random values using the provided function.

        Parameters:
        func (function): The function to use for random initialization. Defaults to numpy.random.normal.
        """
        self.reset()
        for index in self.mutateable_layers:
            if self.layers[index].mutateable_weights:
                self.layers[index].weights += func(
                    size=np.shape(self.layers[index].weights)
                )
            if self.layers[index].mutateable_biases:
                self.layers[index].biases += func(
                    size=np.shape(self.layers[index].biases)
                )

    def save_to_file(self, filename: str):
        """
        Saves the neural network to a file.

        Parameters:
        filename (str): The name of the file to save the network to. Ensure to include the '.pkl' extension.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def __str__(self):
        """
        Returns a string representation of the layers in the network.

        Returns:
        str: Information about the layers in the network.
        """
        display = ""
        for layer in self.layers:
            display = display + "\n---------- \n" + layer.__str__()

        return display + "\n---------- \n"


class layer_dense:
    """
    A dense neural layer.
    """

    def __init__(self, n_in: int, n_out: int):
        """
        Initializes a dense neural layer.

        Parameters:
        n_in (int): The number of input units.
        n_out (int): The number of output units.
        """
        self.biases = np.zeros(n_out)
        self.weights = np.zeros((n_in, n_out))
        self.n_in = n_in
        self.n_out = n_out
        self.mutateable_weights = True
        self.mutateable_biases = True

    def forward(self, X):
        """
        Propagates the input through the layer.

        Parameters:
        X: Input data.

        Returns:
        numpy.ndarray: The output of the layer after the forward pass.
        """
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def reset(self):
        """
        Resets the weights and biases of the layer to 0.
        """
        self.biases = np.zeros(self.n_out)
        self.weights = np.zeros((self.n_in, self.n_out))

    def __str__(self):  # prints info regarding the layer
        """
        Returns a string representation of the layer.

        Returns:
        str: Info
        """
        return (
            self.__class__.__name__
            + "\n"
            + "Weights: "
            + "\n"
            + str(self.weights)
            + "\n"
            + "Biases: "
            + "\n"
            + str(self.biases)
        )


class layer_one_to_one:  # a one to one layer
    def __init__(self, n_in_out: int):
        # initialise weights and biases to 0
        self.n_in = n_in_out
        self.n_out = n_in_out
        self.biases = np.zeros(n_in_out)
        self.weights = np.zeros((n_in_out))
        self.mutateable_weights = True
        self.mutateable_biases = True

    def forward(self, X):
        # push values through the layer
        self.output = np.multiply(X, self.weights) + self.biases
        return self.output

    def reset(self):
        self.biases = np.zeros(self.n_in)
        self.weights = np.zeros((self.n_in))

    def __str__(self):  # prints info regarding the layer
        return (
            self.__class__.__name__
            + "\n"
            + "Weights: "
            + "\n"
            + str(self.weights)
            + "\n"
            + "Biases: "
            + "\n"
            + str(self.biases)
        )


class layer_dropout:  # a dropout layer
    def __init__(self, n_in_out: int, prob: float):
        if prob < 0 or prob > 1:
            raise ValueError("Probability must lie between 0 and 1")
        self.n_in = n_in_out
        self.n_out = n_in_out
        self.weights = np.random.choice([0, 1], size=n_in_out, p=[1 - prob, prob])
        self.prob = prob
        self.mutateable_weights = False
        self.mutateable_biases = False

    def forward(self, X):
        # push values through the layer
        self.weights = np.random.choice(
            [0, 1], size=self.n_in, p=[1 - self.prob, self.prob]
        )  # create random dropout layer each time
        self.output = np.multiply(X, self.weights)
        return self.output

    def reset(self):
        self.weights = np.random.choice(
            [0, 1], size=self.n_in, p=[1 - self.prob, self.prob]
        )

    def __str__(self):  # prints info regarding the layer
        return self.__class__.__name__ + "\n" + "Rate: " + "\n" + str(self.prob)


class layer_1dconv:
    def __init__(self, n_in_out: int, kernel_size: int):
        if n_in_out < kernel_size:
            raise ValueError("Kernel Size must be greater than n_in")
        self.n_in = n_in_out
        self.n_out = n_in_out
        self.kernel_size = kernel_size
        self.left_size = np.floor(kernel_size / 2)
        self.right_size = np.ceil(kernel_size / 2)
        self.mutateable_weights = False
        self.mutateable_biases = False

    def forward(self, X):
        new_X = X.copy()
        for i, row in enumerate(X):
            for j in range(self.n_in):
                left_index = int(max(0, j - self.left_size))
                right_index = int(min(self.n_in, j + self.right_size))
                new_X[i][j] = np.average(row[left_index:right_index])

        self.output = new_X
        return self.output

    def reset(self):
        pass

    def __str__(self):  # prints info regarding the layer
        return (
            self.__class__.__name__
            + "\n"
            + "Kernel Size: "
            + "\n"
            + str(self.kernel_size)
        )


class layer_taylor_features:
    def __init__(self, n_in: int, order: int):
        self.n_in = n_in
        self.n_out = n_in * order
        self.order = order
        self.mutateable_weights = False
        self.mutateable_biases = False

    def forward(self, X):
        new_X = np.zeros((np.shape(X)[0], self.n_out))
        for j in range(len(X)):
            for i in range(self.order):
                new_X[j][(i) * self.n_in : (i + 1) * self.n_in] = X ** (i + 1)

        self.output = new_X
        return self.output

    def reset(self):
        pass

    def __str__(self):  # prints info regarding the layer
        return self.__class__.__name__ + "\n" + "Order: " + "\n" + str(self.order)


class layer_fourier_features:
    def __init__(self, n_in: int):
        self.n_in = n_in
        self.n_out = n_in * 4
        self.mutateable_weights = False
        self.mutateable_biases = False

    def forward(self, X):
        new_X = np.zeros((np.shape(X)[0], self.n_out))

        for j in range(len(X)):
            new_X[j][0 : self.n_in] = np.sin(X)
            new_X[j][self.n_in : 2 * self.n_in] = np.cos(X)
            new_X[j][2 * self.n_in : 3 * self.n_in] = np.sin(2 * X)
            new_X[j][3 * self.n_in : 4 * self.n_in] = np.cos(2 * X)

        self.output = new_X
        return self.output

    def reset(self):
        pass

    def __str__(self):  # prints info regarding the layer
        return self.__class__.__name__


class activation_function:
    def __init__(self, function):
        if callable(function):
            self.function = function
        else:
            raise Exception("Input is not a function")
        self.mutateable_weights = False
        self.mutateable_biases = False

    def forward(self, X):
        self.output = np.apply_along_axis(self.function, 1, X)
        return self.output

    def __str__(self):
        return self.__class__.__name__ + "\n" + self.function.__name__


class relu(activation_function):
    def __init__(self):
        super().__init__(self.function)

    def function(self, x):
        return np.maximum(0, x)


class softmax(activation_function):
    def __init__(self):
        super().__init__(self.function)

    def function(self, x):
        return np.exp(x) / np.sum(np.exp(x))


class sigmoid(activation_function):
    def __init__(self):
        super().__init__(self.function)

    def function(self, x):
        return 1 / (1 + np.exp(-x))


def load_network_from_file(filename):
    with open(filename, "rb") as file:
        loaded_network = pickle.load(file)
    return loaded_network
