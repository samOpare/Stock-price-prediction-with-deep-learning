from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Model:
    """
    This is an class that should not be instantiated!!!
    It is a parent class for each model and is used to define data percentages and common methods
    """

    def __init__(self):
        self.validation_ratio = 0.2
        self.test_ratio = 0.2
        self.train_ratio = 1 - (self.validation_ratio + self.test_ratio)

    def get_train_test(self, data: pd.DataFrame, shuffle=True):
        """
        Devides the given dataframe into training and test data
        :param data: dataframe that contains all data
        :param shuffle: Default: True. If set to true shuffles data
        :return: a list of dataframes where the result are [train, test]
        """
        train, test = train_test_split(data, test_size=self.test_ratio, shuffle=shuffle)
        return train, test

    def get_train_test_separated(self, data: pd.DataFrame, targets: pd.DataFrame, shuffle=True):
        """
        Devides the given dataframe into training and test data
        :param data: the date
        :param targets: the targets
        :param shuffle: shuffle default true
        :return:
        """
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, shuffle=shuffle)
        return train_data, test_data, train_targets, test_targets


class FFNN(Model):
    """
    This class implements a simple feed forward neural network
    with some tweakable parameters.
    """

    def __init__(self, activation_function: str = "relu"):
        """
        Initializes a Feed Forward Neural Network
        """
        super().__init__()
        self.hidden_layer_sizes = (1000, 100, 100)
        self.activation_function = activation_function
        self.solver = 'adam'
        self.alpha = 0.00001  # L2 penalty
        self.batch_size = 'auto'
        self.learning_rate = 'adaptive'
        self.learning_rate_init = 0.0001
        self.power_t = 0.5  # The exponent for inverse scaling learning rate
        self.max_iter = 5000  # number of maximum epochs to run solver instead of running to convergence
        self.shuffle = True  # shuffle samples each iteration
        self.random_state = None  # State of the random number generator
        self.tol = 0.0001  # Tolerance of the optimization when targeting convergence intead of maxiter
        self.verbose = True
        self.warm_start = False
        self.momentum = 0.9  # Not used with adam solver
        self.nesterovs_momentum = True
        self.early_stopping = True
        self.validation_fraction = 0.1
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-08
        self.n_iter_no_change = 20

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation_function,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change)
        if activation_function == "tanh":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the dataframe given by x and uses the series
        given by y as the target
        :param x: the dataframe to train
        :param y: the target values as a series
        :return: nothing
        """
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.model.fit(x, y)
        print("The network trained until reached a training loss of %f in %d iterations!" % (
        self.model.loss_, self.model.n_iter_))

    def predict(self, x: pd.Series):
        """
        Predicts the next days close value after a series X
        :param x: a pandas series
        :return: the close value of the next day
        """
        # return self.model.predict(x.values.reshape(1, -1))
        return self.model.predict(self.scaler.transform(x))


class RNN:
    """
    This class should implement a recurrent neural network
    with an interface similar to FFNN.
    """

    def __init__(self):
        print("TODO")
