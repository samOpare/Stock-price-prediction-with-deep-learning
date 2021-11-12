from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd


class Evaluator:
    """
    This class implements evaluation methods for predicted data
    """

    def __init__(self):
        self.mean_squared_error = 0

    def evaluate(self, predictions, true_values):
        """
        Does Evaluation of the given predictions compared to the given real data
        This should also do some plotting later on.
        :param predictions: a list or series of predictions
        :param true_values: a list or series of original dat
        :param year as string
        """
        # true_values = true_values[year]
        # predictions = predictions[year]
        self.mse(true_values, predictions)
        self.rmse(true_values, predictions)
        self.mean_absolute_percentage_error(true_values, predictions)
        self.plot(true_values, predictions)

    def mse(self, true_values, predictions):
        """
        Calculates the means squared error
        :param: true_values the true values of the data
        :param: predictions the predictions for the data
        :return: the mse
        """
        self.mean_squared_error = mean_squared_error(true_values, predictions)
        print("Least squared error: %f" % self.mean_squared_error)
        return self.mean_squared_error

    def plot(self, true_values, prediction):
        """
        Takes values and predicton and a year and plots these
        :param true_values: the real value
        :param prediction: the prediction
        :param period: the year as pandas period
        """
        ax = true_values.plot(figsize=(15, 6))
        prediction.plot()
        ax.legend()

    def rmse(self, true_values, prediction):
        """
        Calculates the root means squared error
        :param: true_values the true values of the data
        :param: predictions the predictions for the data
        :return: the rmse
        """
        rmse = sqrt(mean_squared_error(true_values, prediction))
        print("Root mean squared error: %f" % rmse)
        return rmse

    def mean_absolute_percentage_error(self, true_values, prediction):
        """
        Calculate the mean absolute percentage error
        :param: true_values the true values of the data
        :param: predictions the predictions for the data
        """
        true_values = np.array(true_values)
        prediction = np.array(prediction)
        mape = np.mean(np.abs((true_values - prediction) / true_values)) * 100
        print("Mean absolute percentage error mean squared error: %f" % mape)
        return mape
