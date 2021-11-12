import pandas as pd


def stochastic_oscillator(res_df_values, company: str):
    """
    Does calculations for Stochastic Oscillators %K and %D
    :param predictions: a list or series of predictions
    :param real_data: a list or series of original data
    :return: the least squared error
    """

    # Create the "L14" column in the DataFrame
    res_df_values['L14'] = res_df_values[company + '_Low'].rolling(window=14).min().bfill()

    # Create the "H14" column in the DataFrame
    res_df_values['H14'] = res_df_values[company + '_High'].rolling(window=14).max().bfill()

    # Create the "%K" column in the DataFrame
    res_df_values['SO%K'] = 100 * (
            (res_df_values[company + '_Close'] - res_df_values['L14']) / (res_df_values['H14'] - res_df_values['L14']))

    # Create the "%D" column in the DataFrame
    res_df_values['SO%D'] = res_df_values['SO%K'].rolling(window=3).mean().bfill()

    return res_df_values


def larry_william_r(res_df_values, company):
    """Calculate larry william %R for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """

    res_df_values['LW%R'] = 100 * (
            (res_df_values['H14'] - res_df_values[company + '_Close']) / (res_df_values['H14'] - res_df_values['L14']))

    return res_df_values


def relative_strength_index(dff, n):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while dff.index[i] + 1 <= dff.index[-1]:
        UpMove = dff['High'][i + 1] - dff['High'][i]
        DoMove = dff['Low'][i] - dff['Low'][i + 1]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    dff = dff.join(RSI)
    return dff
