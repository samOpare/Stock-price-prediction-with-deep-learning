import pandas as pd
import numpy as np


def feature_selection(res_df_values: pd.DataFrame, res_df_targets: pd.DataFrame, coeff_threshold: float = 0.4, uniq_threshold: float = 0.4):
    """
    Feature selection function. Function returns 2 data frames. each one is a result of correlation with one target

    :param res_df_values: the values as a dataframe
    :param res_df_targets: the targets as a dataframe (can include a second target!)
    :param coeff_threshold: threshold to what columns should be kept correlation
    :param uniq_threshold: not used at the moment! Drop all nominal
    :return: df_target1, df_target2 each dataframe contains all data and the target
    """
    df_target1 = pd.DataFrame()
    df_target2 = pd.DataFrame()

    # drop nominal fields from data if unique values < 10
    string_df = res_df_values.select_dtypes(exclude=['integer', 'float'])
    for column in string_df:
        # if (len(res_df_values[column].unique()) < uniq_threshold):
        res_df_values = res_df_values.drop(column, axis=1)

    numeric_df = res_df_values.select_dtypes(include=['integer', 'float'])

    for column in res_df_values:
        corr = numeric_df[column].corr(res_df_targets.iloc[:, 0])
        # print('correlation between ', column, 'and target is ', corr)

        # keeping only columns that have correlation with target higher than threshold
        if (corr < coeff_threshold):
            df_target1 = res_df_values.drop(column, axis=1)
            df_target1['Target'] = res_df_targets.iloc[:, 0]

    # If there is only one target skip
    if res_df_targets.shape[1] == 2:
        for column in res_df_values:  # for i in range(len(res_df_targets.columns)):
            corr = numeric_df[column].corr(res_df_targets.iloc[:, 1])
            # print('correlation between ', column, 'and target is ', corr)

            # keeping only columns that have correlation with target higher than threshold
            if (corr < coeff_threshold):
                df_target2 = res_df_values.drop(column, axis=1)
                df_target2['Target'] = res_df_targets.iloc[:, 1]

    return df_target1, df_target2
    
    
    
    
    
#Alternate feature selection function

# Load libraries
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model
import warnings


def feature_selectionAlt(res_df_values,res_df_targets):
    
    """Performs Recursive Feature Selection

    :param df: pandas.DataFrame
    :param df: pandas.DataFrame
    :return: numpy.Array
    """

    #Suppress an annoying but harmless warning
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # Create a linear regression
    ols = linear_model.LinearRegression()

    # Create recursive feature eliminator that scores features by mean squared errors
    rfecv = RFECV(estimator=ols, step=1, scoring='neg_mean_squared_error')

    # Fit recursive feature eliminator
    rfecv.fit(res_df_values, res_df_targets)

    # Recursive feature elimination
    res_df_values = rfecv.transform(res_df_values)

    # Number of best features
    rfecv.n_features_

    return res_df_values,res_df_targets

