"""
This file defines the main method.
"""
from model import FFNN
from dataloader import DataLoader
from evaluator import Evaluator
import pandas as pd

def main():
    data_loader = DataLoader()
    data = data_loader.load_data()

    ffnn = FFNN()
    train, test = ffnn.get_train_test(data)
    print("Training data excerpt: ")
    print(train.head())

    print("Test data excerpt: ")
    print(test.head())
    ffnn.fit(train.drop('Target', axis=1), train['Target'])
    predictions = ffnn.predict(test.drop(['Target'], axis=1))
    real_values = test['Target']

    print(predictions)

    evaluator = Evaluator()
    evaluator.evaluate(predictions, real_values)
    return 0


__main__ = main()
